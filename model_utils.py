import os
import yaml
import torch
import numpy as np
from pathlib import Path

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.configs.method_configs import all_methods

from camera_utils import update_camera
from general_utils import particle2pose, image_mse


def setup_pipeline_from_config(config_path: Path, device: torch.device = torch.device("cuda")) -> Pipeline:
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    config.load_dir = config.get_checkpoint_dir()

    pipeline = config.pipeline.setup(device=device, test_mode="inference")
    assert isinstance(pipeline, Pipeline)

    pipeline.eval()

    assert config.load_dir is not None
    if config.load_step is None:
        print("Loading latest checkpoint from load_dir")
        if not os.path.exists(config.load_dir):
            raise ValueError(f"{config.load_dir} does not exist")
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    else:
        load_step = config.load_step
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"

    loaded_state = torch.load(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    print(f"Loaded checkpoint from {load_path}")

    return pipeline


def get_losses(pipeline, camera, particles, gt_img):
    """
        Takes in a list of particles and returns the loss for each particle
        based on the photometric loss function.

    Args:
        pipeline (Pipeline): The pipeline object
        camera (Camera): The camera object already initialized with camera_type,
                         w,h,fx,fy,cx,cy,k1,k2,p1,p2. ASSUME THAT THE DOWNSAMPLING
                         FACTOR HAS BEEN INCORPORATED INTO THE CAMERA OBJECT
        particles (list): List of particles containing pose information
        gt_img (np.array): Ground truth image (H,W,3). ASSUME THAT THE IMAGE
                           HAS BEEN DOWNSAMPLED AND IS IN THE RANGE [0, 1]
        downsample (int): Downsample factor for the image and the reconstruction
    """
    num_particles = particles.shape[0]
    losses = np.zeros(num_particles)
    with torch.no_grad():
        for i in range(num_particles):
            camera = update_camera(camera, particle2pose(particles[i]))
            rgb_image = pipeline.model.get_rgba_image(
                                pipeline.model.get_outputs_for_camera(camera)
                        )[..., :3] # This image is in the range [0, 1]
            losses[i] = image_mse(rgb_image, gt_img)
    return losses


def render_camera_pose(pipeline, camera, particle):
    with torch.no_grad():
        camera = update_camera(camera, particle2pose(particle))
        rgb_image = pipeline.model.get_rgba_image(
                            pipeline.model.get_outputs_for_camera(camera)
                    )[..., :3]
    return (rgb_image * 255).cpu().numpy().astype(np.uint8)
