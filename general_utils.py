import json
from turtle import down
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
            whereneg = (q[:, 0] < 0)
            q[whereneg, 0] = -q[whereneg, 0]
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


def inverse_quat(q):
    # q: (4,) ndarray
    # Calculate the inverse of the quaternion
    q_inv = q.copy()
    q_inv[1:] = -q_inv[1:]
    q_inv = normalize_quat(q_inv)
    return q_inv


def delta_particle(particle1, particle2):
    # particle1: (7,) ndarray [x, y, z, qw, qx, qy, qz]
    # particle2: (7,) ndarray [x, y, z, qw, qx, qy, qz]
    # Calculate the delta between two particles
    # delta is of shape (7,) dx, dy, dz, dqw, dqx, dqy, dqz
    delta = np.zeros(7)
    delta[:3] = particle2[:3] - particle1[:3]
    delta[3:] = normalize_quat(
                    quaternion.as_float_array(
                        quaternion.from_float_array(inverse_quat(particle1[3:])) *\
                        quaternion.from_float_array(particle2[3:])
                ))
    return delta


def image_mse(image1, image2):
    # image1: (H, W, 3) torch tensor
    # image2: (H, W, 3) torch tensor
    # Calculate the mean squared error between the two images
    return torch.mean((image1 - image2)**2)


def load_pf_json(json_path):
    # json_path: str
    # Load the particle filter parameters from a json file
    with open(json_path, "r") as f:
        pf_params = json.load(f)
    num_particles = pf_params["num_particles"]
    particle_initial_bounds = pf_params["particle_initial_bounds"]
    
    particle_noise_levels = pf_params["particle_noise_levels"]
    particle_noise_levels["position"] = np.array(particle_noise_levels["position"])
    particle_noise_levels["rotation"] = np.array(particle_noise_levels["rotation"])

    downsample = pf_params["downsample"]

    return num_particles, downsample, particle_initial_bounds, particle_noise_levels
