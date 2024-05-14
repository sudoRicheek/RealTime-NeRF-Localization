import json
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager

from general_utils import pose2particle


class CustomDataManager(VanillaDataManager):
    """
    Custom data manager for loading data from the dataset

    This is important to convert from real world poses to aabb NeRF poses.
    """
    def __init__(
        self,
        config,
        device="cpu",
        test_mode="val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)
        self._target = VanillaDataManager
        self.transform = None
        self.scale = None

    def load_json(self, config_path):
        data_transform_path = Path(config_path).parent / "dataparser_transforms.json"
        data_transform = json.load(open(data_transform_path, "r"))
        transform = np.array(data_transform["transform"]) # 3x4
        transform = np.vstack([transform, np.array([0, 0, 0, 1])]) # 4x4
        scale = float(data_transform["scale"])

        self.transform = transform
        self.scale = scale
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.transform is not None:
            item = self.transform @ np.vstack([item, np.array([0, 0, 0, 1])])
        if self.scale is not None:
            item = item * self.scale
        return item
    

class OdometryDataset(Dataset):
    def __init__(self, dataset_path: Path, config_path: Path, gtpose_path: Path,
                 downsample: int = 8, device: str = "cuda"):
        self.gtdata = json.load(open(gtpose_path, "r")) # gt pose path is path to the json file containing gt poses

        self.data = json.load(open(dataset_path / "transforms.json", "r"))
        self.dir = dataset_path
        self.downsample = downsample
        self.device = device

        data_transform_path = Path(config_path).parent / "dataparser_transforms.json"
        data_transform = json.load(open(data_transform_path, "r"))
        self.transform = np.array(data_transform["transform"])
        self.scale = float(data_transform["scale"])

    def __len__(self):
        return len(self.data["frames"])
    
    def __getitem__(self, idx):
        """
        Returns the image and the transform matrix for the idx-th frame
        img as a tensor of shape (H//downsample, W//downsample, 3)
        transform_matrix as a np.array of shape (7,) [x, y, z, q1, q2, q3, q4]
        """
        file_path = self.data["frames"][idx]["file_path"]
        transform_matrix = np.array(self.data["frames"][idx]["transform_matrix"])
        gt_transform_matrix = np.array(self.gtdata["frames"][idx]["transform_matrix"])
        img = Image.open(self.dir / file_path)
        img = img.resize((img.width // self.downsample, img.height // self.downsample))

        # Normalize the image to [0, 1]
        img = np.array(img) / 255.0
        img = torch.tensor(img).to(self.device)

        transform_matrix = self.transform @ transform_matrix
        gt_transform_matrix = self.transform @ gt_transform_matrix
        transform_matrix[:3, 3] *= self.scale
        gt_transform_matrix[:3, 3] *= self.scale

        return img, pose2particle(transform_matrix[:3]), pose2particle(gt_transform_matrix[:3])
