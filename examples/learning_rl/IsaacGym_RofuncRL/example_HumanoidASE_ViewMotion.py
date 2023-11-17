"""
Humanoid Motion View (RofuncRL)
===========================

Preview the motion of the digital humanoid
"""

import isaacgym
import argparse

from rofunc.config.utils import omegaconf_to_dict, get_config, load_view_motion_config
from rofunc.learning.RofuncRL.tasks import Tasks
from rofunc.learning.RofuncRL.trainers import Trainers


def inference(custom_args):
    view_motion_config = load_view_motion_config(custom_args.config_name)
    task_name = "HumanoidViewMotion"
    args_overrides = [
        f"task={task_name}",
        "train=HumanoidViewMotionASERofuncRL",
        f"device_id=0",
        f"rl_device=cuda:{gpu_id}",
        "headless={}".format(False),
        "num_envs={}".format(16),
    ]
    cfg = get_config("./learning/rl", "config", args=args_overrides)
    cfg.task.env.motion_file = custom_args.motion_file

    # Overwrite
    cfg.task.env.asset.assetFileName = view_motion_config["asset_name"]
    cfg.task.env.asset.assetBodyNum = view_motion_config["asset_body_num"]
    cfg.task.env.asset.assetJointNum = view_motion_config["asset_joint_num"]

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
    trainer = Trainers().trainer_map["ase"](cfg=cfg.train,
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
    parser.add_argument("--config_name", type=str, default="HumanoidSpoonPanSimple")
    parser.add_argument("--motion_file", type=str, default="../hotu/024_amp_3.npy")
    custom_args = parser.parse_args()

    inference(custom_args)
