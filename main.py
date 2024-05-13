import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

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
                    help="Path to the directory containing the GT poses")

# Model Options
parser.add_argument("--pf_config", type=str, default="configs/particle_spot.json",
                    help="Path to the particle filter config file")

# Saving Options
parser.add_argument("--save_nerf_renderings", action="store_true",
                    help="Save the NeRF renderings for the particles")

# Compute Device Options
parser.add_argument("--device", type=str, default="cuda",
                    help="Device to use (cuda/cpu)")

args = parser.parse_args()


def main():
    device = torch.device(args.device)
    data_path = Path(args.load_dir)
    config_path = Path(args.config)
    num_particles, downsample, particle_initial_bounds, particle_noise_levels = load_pf_json(args.pf_config)

    #! Load the data manager
    data_manager = OdometryDataset(data_path, config_path,
                                   downsample=downsample, device=device)

    #! Load the pipeline from the config file. This is our NeRF model + Data Manager
    pipeline = setup_pipeline_from_config(config_path, device)

    #! Load the transforms.json to get the camera parameters
    transforms = json.load(open(data_path / "transforms.json", "r"))
    camera = init_camera_from_json(transforms, downsample=downsample, device=device)

    #! Initialize the particle filter with reasonable bounds on the scene
    PF = ParticleFilter(num_particles, particle_initial_bounds)

    img0, particle0 = data_manager[0]
    for i in tqdm(range(1, len(data_manager))):
        img1, particle1 = data_manager[i]
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

        print(f"GT Pose: {particle2pose(particle1)}")

        cv2.imwrite(f"outputs/spot_outdoor/particle_{i}.png", render_camera_pose(pipeline, camera, best_particle))
        cv2.imwrite(f"outputs/spot_outdoor/gt_{i}.png", (img1 * 255).cpu().numpy().astype(np.uint8))
        cv2.imwrite(f"outputs/spot_outdoor/gt_pose_{i}.png", render_camera_pose(pipeline, camera, particle1))

        img0, particle0 = img1, particle1



if __name__ == "__main__":
    main()
