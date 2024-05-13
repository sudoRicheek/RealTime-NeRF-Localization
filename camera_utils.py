import torch
import numpy as np

from nerfstudio.cameras.cameras import Cameras


def setup_camera(c2w, fx, fy, cx, cy, w, h,
                 k1, k2, p1, p2, camera_type=1,
                 device="cuda"):
    """
    Setup the camera object for the given camera parameters

    Args:
        c2w: Camera to world matrix of shape (3, 4) ndarray
        fx: Focal length in x direction (float)
        fy: Focal length in y direction (float)
        cx: Principal point in x direction (float)
        cy: Principal point in y direction (float)
        w: Image width (int)
        h: Image height (int)
        k1: Radial distortion coefficient 1 (float)
        k2: Radial distortion coefficient 2 (float)
        p1: Tangential distortion coefficient 1 (float)
        p2: Tangential distortion coefficient 2 (float)
        camera_type: Camera type (int) (default: 1 -> PERSPECTIVE)
        device: Device to use (str) (default: "cuda")
    """
    c2w = torch.tensor(c2w, dtype=torch.float32, device=device)
    distortion_params = torch.tensor([[k1, k2, 0.0, 0.0, p1, p2]], dtype=torch.float32, device=device)
    fx = torch.tensor(fx, dtype=torch.float32, device=device)
    fy = torch.tensor(fy, dtype=torch.float32, device=device)
    cx = torch.tensor(cx, dtype=torch.float32, device=device)
    cy = torch.tensor(cy, dtype=torch.float32, device=device)
    camera = Cameras(
        camera_to_worlds=c2w.unsqueeze(0),
        fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h, camera_type=camera_type,
        distortion_params=distortion_params
    )
    return camera


def update_camera(camera, c2w):
    """
    Update the camera object with the new camera to world matrix

    Args:
        camera: Camera object
        c2w: Camera to world matrix of shape (3, 4) ndarray
    """
    camera.camera_to_worlds = torch.tensor(c2w, dtype=torch.float32, device=camera.device).unsqueeze(0)
    return camera


def init_camera_from_json(gt_transforms, device="cuda"):
    """
    Initialize the camera object from the camera parameters in the JSON file

    Args:
        gt_transforms: Camera parameters from the JSON file
        device: Device to use (str) (default: "cuda")
    """
    camera_type = 1
    w, h = gt_transforms["w"], gt_transforms["h"]
    fx, fy, cx, cy = gt_transforms["fl_x"], gt_transforms["fl_y"], gt_transforms["cx"], gt_transforms["cy"]
    k1, k2, p1, p2 = gt_transforms["k1"], gt_transforms["k2"], gt_transforms["p1"], gt_transforms["p2"]
    c2w = np.array(gt_transforms["frames"][0]["transform_matrix"])[:3]
    
    camera = setup_camera(c2w, fx, fy, cx, cy, w, h,
                          k1, k2, p1, p2, camera_type, device)
    return camera
