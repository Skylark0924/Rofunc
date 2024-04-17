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
    cfg.task.env.asset.assetFileName = custom_args.asset
    cfg.task.env.keyBodies = custom_args.keyBodies
    cfg.task.env.contactBodies = custom_args.contactBodies
    if custom_args.use_object_motion:
        cfg.task.env.object_motion_file = custom_args.object_motion_file
        cfg.task.env.object_asset.assetFileName = custom_args.object_asset_files
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
    trainer = Trainers().trainer_map["ase"](cfg=cfg,
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
    # Available types of asset file path:
    #  1. mjcf/hotu_humanoid_w_qbhand_no_virtual.xml
    #  2. mjcf/amp_humanoid_sword_shield.xml
    #  3. mjcf/hotu_humanoid.xml
    #  4. mjcf/amp_humanoid.xml
    #  5. mjcf/hotu_humanoid_w_qbhand_full.xml
    #  6. mjcf/UnitreeH1/h1_w_qbhand.xml
    parser.add_argument("--asset", type=str, default="mjcf/UnitreeH1/h1_w_qbhand.xml")
    parser.add_argument("--keyBodies", type=list, default=["right_hand", "left_hand", "right_ankle_link", "left_ankle_link"])
    parser.add_argument("--contactBodies", type=list, default=["right_ankle_link", "left_ankle_link"])
    # Available types of motion file path:
    #  1. test data provided by rofunc: `examples/data/hotu/*.npy`
    #  2. custom motion file with absolute path
    parser.add_argument("--motion_file", type=str, default="/home/ubuntu/Github/Xianova_Robotics/Rofunc-secret/examples/data/hotu2/test_data_01_optitrack2h1.npy")

    parser.add_argument("--use_object_motion", action="store_true")
    parser.add_argument("--object_motion_file", type=str, default="examples/data/hotu2/test_data_04_optitrack.csv")
    # parser.add_argument("--object_asset_names", type=str, default=["box:marker 001", "box:marker 002", "box:marker 003", "box:marker 004"])
    parser.add_argument("--object_asset_names", type=str, default=["box"])
    parser.add_argument("--object_asset_files", type=str, default=["mjcf/objects/lab_box.xml"])
    # parser.add_argument("--object_asset_sizes", type=str, default=[[0.05, 0.05, 0.05], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05]])
    parser.add_argument("--object_asset_sizes", type=str, default=[[1, 1, 1]])
    custom_args = parser.parse_args()

    inference(custom_args)
