import numpy as np
from numpy.random import default_rng

import quaternion

from general_utils import normalize_quat

rng = default_rng(42)


class ParticleFilter:
    def __init__(self, num_particles, particle_initial_bounds) -> None:
        # We model particles as position (3,) and rotation (quaternions) (4,)
        self.num_particles = num_particles
        self.weights = np.ones(num_particles)
        self.particle_initial_bounds = particle_initial_bounds # this is a dictionary
        self.init_particles(num_particles)

    def init_particles(self, num_particles):
        positions  = np.random.uniform(self.particle_initial_bounds["position"][0],
                                       self.particle_initial_bounds["position"][1],
                                       (num_particles, 3))
        quaternions = quaternion.as_float_array(
                        quaternion.from_euler_angles(
                            np.random.uniform(self.particle_initial_bounds["rotation"][0],
                                              self.particle_initial_bounds["rotation"][1],
                                              (num_particles, 3))
                            )
                      )
        quaternions = normalize_quat(quaternions)
        self.particles = np.concatenate((positions, quaternions), axis=1) # (num_particles, 7)
    
    def predict_particles(self, del_particle, position_noise, rotation_noise):
        # del_particle: (7,) Delta change in the particle pose
        # position_noise: (3,) nd array of noise in position
        # rotation_noise: (3,) nd array of noise in rotation euler angles
        trans_odom = del_particle[:3]
        quat_odom = quaternion.from_float_array(del_particle[3:]) # q1.inv() * q2
        trans_noise = position_noise * rng.normal(size=(self.num_particles, 3))

        # With this I realize that this quaternion library doesnt have the most basic functionalities.
        # TODO: Use a better quaternion library, or do it on numpy directly.
        quat_noise = quaternion.as_quat_array(
                        normalize_quat(
                            quaternion.as_float_array(
                                quaternion.from_euler_angles(
                                    rotation_noise * rng.normal(size=(self.num_particles, 3))
                     ))))
        # predict particles
        self.particles[:, :3] = self.particles[:, :3] + trans_odom + trans_noise
        for i in range(self.num_particles):
            self.particles[i, 3:] = quaternion.as_float_array(
                                        quaternion.from_float_array(self.particles[i, 3:]) *\
                                        quat_odom * quat_noise[i]
                                    )

    def update_weights(self, losses):
        # Multiply 1/loss by total number of pixels reconstructed or not??
        # losses: (num_particles,) dtype=float64
        # losses are the photometric losses from each particle pose.

        #* Note that we might need to upper bound the weights at some point.
        #* Or multiply them if they are too small.

        #! I am not sure why, but this is what the author of
        #! Monte Carlo NeRF suggested in the paper (raise to the power of 4)
        self.weights = (1/losses) ** 4 
        self.weights = self.weights / np.sum(self.weights)

    def resample_particles(self):
        indices = rng.choice(self.num_particles, size=self.num_particles,
                             p=self.weights, replace=True)
        self.particles = self.particles[indices]

    def get_best_particle(self):
        best_particle = self.particles[np.argmax(self.weights)]
        return best_particle
    
    def get_average_position(self):
        average_position = np.mean(self.particles[:, :3], axis=0)
        return average_position
    
    def get_weighted_average_position(self):
        weighted_average_position = np.average(self.particles[:, :3], axis=0, weights=self.weights)
        return weighted_average_position

    def get_average_quaternion(self):
        A = np.zeros((4, 4))
        for i in range(self.num_particles):
            A += np.outer(self.particles[i, 3:], self.particles[i, 3:])
        A = (1.0/self.num_particles) * A
        eigvals, eigvecs = np.linalg.eig(A)
        eigvecs = eigvecs[:, eigvals.argsort()[::-1]]
        average_quaternion = eigvecs[:, 0]
        average_quaternion = normalize_quat(average_quaternion)
        return average_quaternion

    
    def get_average_rotation(self):
        average_quaternion = self.get_average_quaternion()
        average_rotation = quaternion.as_rotation_matrix(quaternion.from_float_array(average_quaternion))
        return average_rotation


if __name__ == "__main__":
    particle_initial_bounds = {
        "position": [[-5, -5, -5], [5, 5, 5]],
        "rotation": [[-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi]]
    }
    pf = ParticleFilter(100, particle_initial_bounds)
    pf.init_particles(100)
