"""
Humanoid Motion View (RofuncRL)
===========================

Preview the motion of the digital humanoid
"""

import isaacgym
import argparse

import rofunc as rf
from rofunc.config.utils import omegaconf_to_dict, get_config
from rofunc.learning.RofuncRL.tasks import Tasks
from rofunc.learning.RofuncRL.trainers import Trainers


def inference(custom_args):
    task_name = "HumanoidASEViewMotion"
    args_overrides = [
        f"task={task_name}",
        "train=HumanoidASEViewMotionASERofuncRL",
        f"device_id=0",
        f"rl_device=cuda:{gpu_id}",
        "headless={}".format(False),
        "num_envs={}".format(1),
    ]
    cfg = get_config("./learning/rl", "config", args=args_overrides)
    cfg.task.env.motion_file = custom_args.motion_file
    cfg.task.env.asset.assetFileName = custom_args.asset

    cfg_dict = omegaconf_to_dict(cfg.task)

    # Instantiate the Isaac Gym environment
    infer_env = Tasks().task_map[task_name](cfg=cfg_dict,
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
                                            env_name=task_name,
                                            hrl=False,
                                            inference=True)

    # Start inference
    trainer.inference()


if __name__ == "__main__":
    gpu_id = 1

    parser = argparse.ArgumentParser()
    # Available types of asset file path:
    #  1. mjcf/hotu_humanoid_w_qbhand_no_virtual.xml
    #  2. mjcf/amp_humanoid_sword_shield.xml
    #  3. mjcf/hotu/hotu_humanoid.xml
    #  4. mjcf/amp_humanoid.xml
    parser.add_argument("--asset", type=str, default="mjcf/hotu/hotu_humanoid.xml")
    # Available types of motion file path:
    #  1. test data provided by rofunc: `examples/data/amp/*.npy`
    #  2. custom motion file with absolute path
    parser.add_argument("--motion_file", type=str, default=rf.oslab.get_rofunc_path('../examples/data/amp/amp_humanoid_backflip.npy'))
    custom_args = parser.parse_args()

    inference(custom_args)
