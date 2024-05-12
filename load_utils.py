import os
import yaml
import torch
from pathlib import Path

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.configs.method_configs import all_methods


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
