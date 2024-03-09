"""
Humanoid Motion View (RofuncRL)
===========================

Preview the motion of the digital humanoid
"""

import isaacgym
import argparse

from rofunc.config.utils import omegaconf_to_dict, get_config
from rofunc.learning.RofuncRL.tasks import Tasks
from rofunc.learning.RofuncRL.trainers import Trainers


def inference(custom_args):
    args_overrides = [
        f"task={custom_args.task}",
        "train=HumanoidHOTUViewMotionRofuncRL",
        f"device_id=0",
        f"rl_device=cuda:{gpu_id}",
        "headless={}".format(False),
        "num_envs={}".format(1),
    ]
    cfg = get_config("./learning/rl", "config", args=args_overrides)
    cfg.task.env.motion_file = custom_args.motion_file
    cfg.task.env.object_motion_file = custom_args.object_motion_file
    cfg.task.env.object_asset.assetName = custom_args.object_asset_names
    cfg.task.env.object_asset.assetSize = custom_args.object_asset_sizes

    cfg_dict = omegaconf_to_dict(cfg.task)

    # Instantiate the Isaac Gym environment
    infer_env = Tasks().task_map[custom_args.task](cfg=cfg_dict,
                                                   rl_device=cfg.rl_device,
                                                   sim_device=f'cuda:{cfg.device_id}',
                                                   graphics_device_id=cfg.device_id,
                                                   headless=cfg.headless,
                                                   virtual_screen_capture=cfg.capture_video,  # TODO: check
                                                   force_render=cfg.force_render)

    # Instantiate the RL trainer
    trainer = Trainers().trainer_map["hotu"](cfg=cfg,
                                             env=infer_env,
                                             device=cfg.rl_device,
                                             env_name=custom_args.task,
                                             hrl=False,
                                             inference=True)

    # Start inference
    trainer.inference()


if __name__ == "__main__":
    gpu_id = 0

    parser = argparse.ArgumentParser()
    # Find or define your own config in `rofunc/config/`
    parser.add_argument("--task", type=str, default="HumanoidHOTUViewMotion")
    # Available types of motion file path:
    #  1. test data provided by rofunc: `examples/data/hotu/*.npy`
    #  2. custom motion file with absolute path
    parser.add_argument("--motion_file", type=str,
                        default="examples/data/hotu2/test_data_01_optitrack2hotu.npy")
    parser.add_argument("--object_motion_file", type=str,
                        default="examples/data/hotu2/test_data_01_optitrack.csv")
    parser.add_argument("--object_asset_names", type=str, default=["box:marker 001", "box:marker 002", "box:marker 003", "box:marker 004"])
    # parser.add_argument("--object_asset_files", type=str, default=["Box.urdf"])
    parser.add_argument("--object_asset_sizes", type=str, default=[[0.05, 0.05, 0.05], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05]])
    custom_args = parser.parse_args()

    inference(custom_args)
