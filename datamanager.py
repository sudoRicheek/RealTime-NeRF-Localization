import json
import numpy as np
from pathlib import Path

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager

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
    
