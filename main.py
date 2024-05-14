import os
import cv2
import json
import quaternion
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from datamanager import OdometryDataset
from model_utils import setup_pipeline_from_config, get_losses, render_camera_pose
from camera_utils import init_camera_from_json
from particle_filter import ParticleFilter
from general_utils import delta_particle, particle2pose, load_pf_json


parser = argparse.ArgumentParser(usage="Real-Time NeRF Localization")

# Data Options
parser.add_argument("--config", type=str, required=True, default="outputs/spot_outdoor/nerfacto/2024-05-11_205736/config.yml",
                    help="Path to the config file to load")
parser.add_argument("--load_dir", type=str, default="/home/richeek/spot_outdoor",
                    help="Path to the directory containing the COLMAP poses and images")
parser.add_argument("--load_gtpose", type=str,
                    help="Path to the json file containing the GT poses from FasterLIO")

# Model Options
parser.add_argument("--pf_config", type=str, default="configs/particle_spot.json",
                    help="Path to the particle filter config file")

# Saving Options
parser.add_argument("--save_nerf_renderings", action="store_true",
                    help="Save the NeRF renderings for the particles")
parser.add_argument("--output_dir", type=str, default="outputs/spot_outdoor/renderings",
                    help="Directory to save the NeRF renderings")
parser.add_argument("--plot_particles", action="store_true",
                    help="Plot the particles in 2D X-Z plane")

# Compute Device Options
parser.add_argument("--device", type=str, default="cuda",
                    help="Device to use (cuda/cpu)")

args = parser.parse_args()


def main():
    device = torch.device(args.device)
    data_path = Path(args.load_dir)
    config_path = Path(args.config)
    gtpose_path = Path(args.load_gtpose)
    num_particles, downsample, particle_initial_bounds, particle_noise_levels = load_pf_json(args.pf_config)

    os.makedirs(args.output_dir, exist_ok=True)

    #! Load the data manager
    data_manager = OdometryDataset(data_path, config_path, gtpose_path,
                                   downsample=downsample, device=device)

    #! Load the pipeline from the config file. This is our NeRF model + Data Manager
    pipeline = setup_pipeline_from_config(config_path, device)

    #! Load the transforms.json to get the camera parameters
    transforms = json.load(open(data_path / "transforms.json", "r"))
    camera = init_camera_from_json(transforms, downsample=downsample, device=device)

    #! Initialize the particle filter with reasonable bounds on the scene
    PF = ParticleFilter(num_particles, particle_initial_bounds)

    if args.plot_particles:
        gt_xz = []
        particles_xz = []
        best_particle_xz = []

    img0, particle0, _ = data_manager[0]

    euler_angles = quaternion.as_euler_angles(quaternion.from_float_array(particle0[3:]))
    print(f"Initial Pose: {particle0}")
    print(f"Initial Euler Angles: {euler_angles}")

    for i in tqdm(range(1, len(data_manager))):
        img1, particle1, gtparticle1 = data_manager[i]
        del_particle = delta_particle(particle0, particle1)

        # predict the next state
        PF.predict_particles(del_particle, particle_noise_levels["position"], particle_noise_levels["rotation"])

        # update the weights
        losses = get_losses(pipeline, camera, PF.particles, img1)
        PF.update_weights(losses)

        # resample the particles
        PF.resample_particles()

        #! Print Stuff
        best_particle = PF.get_best_particle()
        best_pose = particle2pose(best_particle)
        avg_position = PF.get_average_position()
        avg_rotation = PF.get_average_rotation()

        print(f"Best Particle: {best_particle}")
        print(f"Best Pose: {best_pose}")
        print(f"Average Position: {avg_position}")
        print(f"Average Rotation: {avg_rotation}")

        print(f"GT Pose: {particle2pose(gtparticle1)}")

        cv2.imwrite(f"{args.output_dir}/best_particle_{i}.png", render_camera_pose(pipeline, camera, best_particle))
        cv2.imwrite(f"{args.output_dir}/gt_{i}.png", (img1 * 255).cpu().numpy().astype(np.uint8))
        cv2.imwrite(f"{args.output_dir}/gt_pose_{i}.png", render_camera_pose(pipeline, camera, gtparticle1))

        if args.plot_particles:
            gt_xz.append([gtparticle1[0], gtparticle1[1]])
            particles_xz.append(PF.particles[:, [0, 1]])
            best_particle_xz.append([best_particle[0], best_particle[1]])

            plt.figure()
            plt.scatter(PF.particles[:, 1], PF.particles[:, 0], s=2, label="Particles", c='b')
            plt.scatter([gtparticle1[1]], [gtparticle1[0]], s=7, label="GT", c='r')
            plt.scatter([best_particle[1]], [best_particle[0]], s=7, label="Best Particle", c='g')
            plt.xlim(-1, 1)
            plt.ylim(-1, 1.2)
            plt.legend()
            plt.xlabel("X Right of camera")
            plt.ylabel("Z Front of camera")
            plt.savefig(f"{args.output_dir}/particles_scatter_{i}.png", dpi=400)
            plt.close()

        particle0 = particle1

    if args.plot_particles:
        gt_xz = np.array(gt_xz)
        particles_xz = np.array(particles_xz)
        best_particle_xz = np.array(best_particle_xz)

        plt.scatter(gt_xz[:, 1], gt_xz[:, 0], s=5, label="GT", c='r')
        plt.scatter(best_particle_xz[:, 1], best_particle_xz[:, 0], s=5, label="Best Particle", c='g')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1.2)
        plt.legend()
        plt.xlabel("X Right of camera")
        plt.ylabel("Z Front of camera")
        plt.savefig(f"{args.output_dir}/particles.png", dpi=400)
        plt.close()


if __name__ == "__main__":
    main()
