import cv2
import json
import torch
import argparse
import numpy as np
from pathlib import Path

from nerfstudio.cameras.cameras import Cameras

from datamanager import CustomDataManager
from load_utils import setup_pipeline_from_config
from camera_utils import setup_camera, update_camera



parser = argparse.ArgumentParser(usage="Real-Time NeRF Localization")
parser.add_argument("--config", type=str, required=True, default="outputs/spot_outdoor/nerfacto/2024-05-11_205736/config.yml",
                    help="Path to the config file to load")
parser.add_argument("--load_dir", type=str, default="/home/richeek/spot_outdoor",
                    help="Path to the directory containing the GT poses")

args = parser.parse_args()





def main():
    data_path = Path(args.load_dir)
    config_path = Path(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = setup_pipeline_from_config(config_path, device)

    #! Load the transforms.json
    gt_transforms = json.load(open(data_path / "transforms.json", "r"))
    camera_type = 1
    w, h = gt_transforms["w"], gt_transforms["h"]
    fx, fy, cx, cy = gt_transforms["fl_x"], gt_transforms["fl_y"], gt_transforms["cx"], gt_transforms["cy"]
    k1, k2, p1, p2 = gt_transforms["k1"], gt_transforms["k2"], gt_transforms["p1"], gt_transforms["p2"]
    c2w = np.array(gt_transforms["frames"][0]["transform_matrix"])[:3]
    camera = setup_camera(c2w, fx, fy, cx, cy, w, h,
                          k1, k2, p1, p2, camera_type, device)


    # gt_image = gt_transforms["frames"][50]["file_path"]
    outputs_from_model = pipeline.model.get_outputs_for_camera(camera)["rgb"]

    cv2.imwrite("outputs/nerf1.png", (outputs_from_model.cpu().numpy()*255).astype(np.uint8))

    c2w = np.array(gt_transforms["frames"][51]["transform_matrix"])[:3]
    camera = update_camera(camera, c2w)

    outputs_from_model = pipeline.model.get_outputs_for_camera(camera)["rgb"]

    # print("Time taken in seconds: ", time.time()-start)
    # rgba_image = pipeline.model.get_rgba_image(outputs_from_model)[..., :3]
    # rgba_image = (rgba_image.numpy()*255).astype(np.uint8)

    cv2.imwrite("outputs/nerf2.png", (outputs_from_model.cpu().numpy()*255).astype(np.uint8))






    breakpoint()




if __name__ == "__main__":
    main()
