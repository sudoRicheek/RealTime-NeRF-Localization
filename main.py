import cv2
import json
import torch
import argparse
import numpy as np
from pathlib import Path

from datamanager import CustomDataManager
from model_utils import setup_pipeline_from_config
from camera_utils import init_camera_from_json

parser = argparse.ArgumentParser(usage="Real-Time NeRF Localization")

# Data Options
parser.add_argument("--config", type=str, required=True, default="outputs/spot_outdoor/nerfacto/2024-05-11_205736/config.yml",
                    help="Path to the config file to load")
parser.add_argument("--load_dir", type=str, default="/home/richeek/spot_outdoor",
                    help="Path to the directory containing the GT poses")

# Model Options
parser.add_argument("--num_particles", type=int, default=100,
                    help="Number of particles to use in the particle filter")

# Compute Device Options
parser.add_argument("--device", type=str, default="cuda",
                    help="Device to use (cuda/cpu)")

args = parser.parse_args()


def main():
    device = torch.device(args.device)
    data_path = Path(args.load_dir)
    config_path = Path(args.config)

    pipeline = setup_pipeline_from_config(config_path, device)

    #! Load the transforms.json to get the camera parameters
    gt_transforms = json.load(open(data_path / "transforms.json", "r"))
    camera = init_camera_from_json(gt_transforms, device)

    #! Initialize the particle filter with reasonable bounds on the scene
    particle_initial_bounds = {
        "position": [[-1, -4, -1], [0, -2, 0]],
        "rotation": [[-np.pi/2, np.pi/4, np.pi/4], [-np.pi/4, np.pi/2, 3*np.pi/4]]
    }
    particle_noise_levels = {
        "position": np.array([0.1, 0.1, 0.1]),
        "rotation": np.array([0.01, 0.01, 0.01])
    }
    




    breakpoint()




if __name__ == "__main__":
    main()
