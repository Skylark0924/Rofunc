"""
HOTU Skill Transfer for Humnaoid Robots (RofuncRL)
===========================

Humanoid Object-centric Embodiment Skill Transfer based on Unified Digitial Humanoid Model (HOTU) for Humanoid Robots
"""

import isaacgym
import argparse
from omegaconf import OmegaConf

from rofunc.config.utils import omegaconf_to_dict, get_config, get_view_motion_config
from rofunc.learning.RofuncRL.tasks import Tasks
from rofunc.learning.RofuncRL.trainers import Trainers


def train(custom_args):
    # Config task and trainer parameters for Isaac Gym environments
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}RofuncRL".format(custom_args.task),
                      "device_id={}".format(custom_args.sim_device),
                      "rl_device=cuda:{}".format(custom_args.rl_device),
                      "headless={}".format(custom_args.headless),
                      "num_envs={}".format(custom_args.num_envs)]
    cfg = get_config("./learning/rl", "config", args=args_overrides)
    cfg_view_motion = get_view_motion_config(custom_args.humanoid_robot_type)
    cfg.task.env = OmegaConf.merge(cfg.task.env, cfg_view_motion)
    cfg_dict = omegaconf_to_dict(cfg.task)

    set_seed(cfg.train.Trainer.seed)

    # Instantiate the Isaac Gym environment
    env = Tasks().task_map[custom_args.task](cfg=cfg_dict,
                                             rl_device=cfg.rl_device,
                                             sim_device=f'cuda:{cfg.device_id}',
                                             graphics_device_id=cfg.device_id,
                                             headless=cfg.headless,
                                             virtual_screen_capture=cfg.capture_video,  # TODO: check
                                             force_render=cfg.force_render)

    # Instantiate the RL trainer
    hrl = False if "Getup" in custom_args.task or "Perturb" in custom_args.task or "ViewMotion" in custom_args.task else True
    trainer = Trainers().trainer_map["hotu"](cfg=cfg,
                                             env=env,
                                             device=cfg.rl_device,
                                             env_name=custom_args.task,
                                             hrl=hrl)

    # Start training
    trainer.train()


def inference(custom_args):
    args_overrides = [f"task={custom_args.task}",
                      f"train={custom_args.task}RofuncRL",
                      "device_id=0",
                      f"rl_device=cuda:{gpu_id}",
                      "headless={}".format(False),
                      "num_envs={}".format(16),
                      ]
    cfg = get_config("./learning/rl", "config", args=args_overrides)
    cfg_view_motion = get_view_motion_config(custom_args.humanoid_robot_type)
    cfg.task.env = OmegaConf.merge(cfg.task.env, cfg_view_motion)
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
    parser.add_argument("--task", type=str, default="HumanoidHOTUGetup")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--sim_device", type=int, default=0)
    parser.add_argument("--rl_device", type=int, default=gpu_id)

    # Available types of asset file path:
    #  1. HOTUHumanoid
    #  2. HOTUHumanoidWQbhand
    #  3. HOTUH1WQbhand
    #  4. HOTUCURIWQbhand
    #  5. HOTUWalker
    #  6. HOTUBruce
    #  7. HOTUZJUHumanoid
    parser.add_argument("--humanoid_robot_type", type=str, default="HOTUH1WQbhand")

    parser.add_argument("--headless", type=str, default="True")
    parser.add_argument("--inference", action="store_true", help="turn to inference mode while adding this argument")
    parser.add_argument("--ckpt_path", type=str, default=None)

    custom_args = parser.parse_args()

    if not custom_args.inference:
        train(custom_args)
    else:
        inference(custom_args)

