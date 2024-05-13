import torch
import numpy as np
import quaternion


def normalize_quat(q):
    # q: (4,) ndarray
    # or q: (N, 4) ndarray
    # or q: Quaternion object
    # Normalize the quaternion
    if isinstance(q, np.ndarray):
        if q.ndim == 1:
            if q[0] < 0:
                q = -q
            q = q / np.linalg.norm(q)
        elif q.ndim == 2:
            whereneg = q[:, 0] < 0
            q[whereneg] = -q[whereneg]
            q = q / np.linalg.norm(q, axis=1)[:, None]
    return q

def pose2particle(pose):
    # pose: (3, 4) R | t
    # Convert the pose matrix to a particle
    # particle is of shape (7,) x, y, z, qx, qy, qz, qw
    particle = np.zeros(7)
    particle[:3] = pose[:, 3]
    particle[3:] = normalize_quat(
                        quaternion.as_float_array(
                            quaternion.from_rotation_matrix(
                                pose[:, :3]
                   )))
    return particle


def particle2pose(particle):
    # particle: (7,) ndarray
    # Convert the particle to a pose matrix
    # pose matrix is of shape (3, 4) R | t
    pose = np.zeros((3, 4))
    pose[:, :3] = quaternion.as_rotation_matrix(
                        quaternion.from_float_array(
                            normalize_quat(particle[3:])
                        )
                   )
    pose[:, 3] = particle[:3]
    return pose


def image_mse(image1, image2):
    # image1: (H, W, 3) torch tensor
    # image2: (H, W, 3) torch tensor
    # Calculate the mean squared error between the two images
    return torch.mean((image1 - image2)**2)
