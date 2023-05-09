import os

import tensorflow as tf
import numpy as np

from scipy.spatial.transform import Rotation as R
import tensorflow.keras as keras


def load_motion(path):
    motion = np.load(path, mmap_mode='r')

    reduce_factor = int(motion['mocap_framerate'] // 30)
    pose = motion['poses'][::reduce_factor, :72]
    trans = motion['trans'][::reduce_factor, :]

    separate_arms(pose)

    # Swap axes
    swap_rotation = R.from_euler('zx', [-90, 270], degrees=True)
    root_rot = R.from_rotvec(pose[:, :3])
    pose[:, :3] = (swap_rotation * root_rot).as_rotvec()
    trans = swap_rotation.apply(trans)

    # Center model in first frame
    trans = trans - trans[0] 

    # Compute velocities
    trans_vel = finite_diff(trans, 1 / 30)

    return pose.astype(np.float32), trans.astype(np.float32), trans_vel.astype(np.float32)


def separate_arms(poses, angle=20, left_arm=17, right_arm=16):
    num_joints = poses.shape[-1] //3

    poses = poses.reshape((-1, num_joints, 3))
    rot = R.from_euler('z', -angle, degrees=True)
    poses[:, left_arm] = (rot * R.from_rotvec(poses[:, left_arm])).as_rotvec()
    rot = R.from_euler('z', angle, degrees=True)
    poses[:, right_arm] = (rot * R.from_rotvec(poses[:, right_arm])).as_rotvec()

    poses[:, 23] *= 0.1
    poses[:, 22] *= 0.1

    return poses.reshape((poses.shape[0], -1))


def finite_diff(x, h, diff=1):
    if diff == 0:
        return x

    v = np.zeros(x.shape, dtype=x.dtype)
    v[1:] = (x[1:] - x[0:-1]) / h

    return finite_diff(v, h, diff-1)


class FaceNormals(keras.layers.Layer):
    def __init__(self, normalize=True, **kwargs):
        super(FaceNormals, self).__init__(**kwargs)
        self.normalize = normalize

    def call(self, vertices, faces):
        v = vertices
        f = faces

        if v.shape.ndims == (f.shape.ndims + 1):
            f = tf.tile([f], [tf.shape(v)[0], 1, 1])   

        # Warning: tf.gather is prone to memory problems
        triangles = tf.gather(v, f, axis=-2, batch_dims=v.shape.ndims - 2) 

        # Compute face normals
        v0, v1, v2 = tf.unstack(triangles, axis=-2)
        e1 = v0 - v1
        e2 = v2 - v1
        face_normals = tf.linalg.cross(e2, e1) 

        if self.normalize:
            face_normals = tf.math.l2_normalize(face_normals, axis=-1)

        return face_normals


class PairwiseDistance(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PairwiseDistance, self).__init__(**kwargs)

    def call(self, A, B):
        rA = tf.reduce_sum(tf.square(A), axis=-1)
        rB = tf.reduce_sum(tf.square(B), axis=-1)
        transpose_axes = [0, 2, 1] 
        distances = - 2*tf.matmul(A, tf.transpose(B, transpose_axes)) + rA[:, :, tf.newaxis] + rB[:, tf.newaxis, :]
        return distances


class NearestNeighbour(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NearestNeighbour, self).__init__(**kwargs)

    def call(self, A, B):
        distances = PairwiseDistance(dtype=self.dtype)(A, B)
        nearest_neighbour = tf.argmin(distances, axis=-1)
        return tf.cast(nearest_neighbour, dtype=tf.int32)