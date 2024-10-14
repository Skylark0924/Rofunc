"""
HOTU Skill Transfer for Humnaoid Robots (RofuncRL)
===========================

Humanoid Object-centric Embodiment Skill Transfer based on Unified Digitial Humanoid Model (HOTU) for Humanoid Robots
"""

import isaacgym
import argparse
from omegaconf import OmegaConf

from rofunc.config.utils import get_view_motion_config
from rofunc.config.utils import omegaconf_to_dict, get_config
from rofunc.learning.RofuncRL.tasks import Tasks
from rofunc.learning.RofuncRL.trainers import Trainers
from rofunc.learning.utils.utils import set_seed


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
    cfg.task.env = OmegaConf.merge(cfg.task.env, cfg_view_motion["env"])
    cfg.task.task = OmegaConf.merge(cfg.task.task, cfg_view_motion["task"])
    if custom_args.debug == "True":
        cfg.train.Trainer.wandb = False
    if custom_args.mode == "HRL":
        cfg.train.Agent.llc_ckpt_path = custom_args.llc_ckpt_path
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
    trainer = Trainers().trainer_map["hotu"](cfg=cfg,
                                             env=env,
                                             device=cfg.rl_device,
                                             env_name=f"{custom_args.task}_{custom_args.humanoid_robot_type}",
                                             mode=custom_args.mode)

    if custom_args.ckpt_path is not None:
        # load checkpoint
        trainer.agent.load_ckpt(custom_args.ckpt_path,)
                                # load_modules=["policy", "value", "optimizer_policy", "optimizer_value",
                                #               "state_preprocessor", "value_preprocessor"])
                                # load_modules=["policy", "value", "optimizer_policy", "optimizer_value",
                                #               "state_preprocessor", "value_preprocessor", "encoder", "discriminator"])
        # load_modules=["encoder", "discriminator"])
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
    cfg.task.env = OmegaConf.merge(cfg.task.env, cfg_view_motion["env"])
    cfg.task.task = OmegaConf.merge(cfg.task.task, cfg_view_motion["task"])
    if custom_args.mode == "HRL":
        cfg.train.Agent.llc_ckpt_path = custom_args.llc_ckpt_path
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
                                             env_name=f"{custom_args.task}_{custom_args.humanoid_robot_type}",
                                             mode=custom_args.mode,
                                             inference=True)

    trainer.agent.load_ckpt(custom_args.ckpt_path)
    # Start inference
    trainer.inference()


if __name__ == "__main__":
    gpu_id = 0

    parser = argparse.ArgumentParser()
    # Available tasks:
    #  1. HumanoidHOTUGetup
    #  2. HumanoidHOTUViewMotion
    #  3. HumanoidHOTUPerturb
    #  4. HumanoidHOTUHeading
    #  5. HumanoidHOTULocation
    #  6. HumanoidHOTUStyle
    parser.add_argument("--task", type=str, default="HumanoidHOTUPerturb")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--sim_device", type=int, default=gpu_id)
    parser.add_argument("--rl_device", type=int, default=gpu_id)

    # Available types of asset file path:
    #  1. HOTUHumanoid
    #  2. HOTUHumanoidWQbhandNew
    #  3. HOTUH1WQbhandNew
    #  4. HOTUCURIWQbhand
    #  5. HOTUWalker
    #  6. HOTUBruce
    #  7. HOTUZJUHumanoid
    #  8. HOTUZJUHumanoidWQbhandNew
    parser.add_argument("--humanoid_robot_type", type=str, default="HOTUBruce")
    # Available modes:
    #  1. LLC
    #  2. HRL
    parser.add_argument("--mode", type=str, default="LLC")

    parser.add_argument("--debug", type=str, default="False")
    parser.add_argument("--headless", type=str, default="True")
    parser.add_argument("--inference", action="store_false", help="turn to inference mode while adding this argument")
    parser.add_argument("--ckpt_path", type=str, default="../examples/learning_rl/IsaacGym_RofuncRL/saved_runs/RofuncRL_HOTUTrainer_HumanoidHOTUGetup_HOTUBruce_24-05-28_13-51-39-584325_body_amp5/checkpoints/best_ckpt.pth")

    # HOTU
    # parser.add_argument("--llc_ckpt_path", type=str, default="../examples/learning_rl/IsaacGym_RofuncRL/saved_runs/RofuncRL_HOTUTrainer_HumanoidHOTUGetup_HOTUHumanoidWQbhandNew_24-05-26_21-16-24-361269_body_amp5/checkpoints/best_ckpt.pth")
    # ZJU
    # parser.add_argument("--llc_ckpt_path", type=str, default="../examples/learning_rl/IsaacGym_RofuncRL/saved_runs/RofuncRL_HOTUTrainer_HumanoidHOTUGetup_HOTUZJUHumanoidWQbhandNew_24-05-26_18-57-20-244370_body_amp5/checkpoints/best_ckpt.pth")
    # H1
    # parser.add_argument("--llc_ckpt_path", type=str, default="../examples/learning_rl/IsaacGym_RofuncRL/saved_runs/RofuncRL_HOTUTrainer_HumanoidHOTUGetup_HOTUH1WQbhandNew_24-05-27_16-59-15-598225_body_amp5/checkpoints/best_ckpt.pth")
    # Bruce
    parser.add_argument("--llc_ckpt_path", type=str, default="../examples/learning_rl/IsaacGym_RofuncRL/saved_runs/RofuncRL_HOTUTrainer_HumanoidHOTUGetup_HOTUBruce_24-05-28_13-51-39-584325_body_amp5/checkpoints/best_ckpt.pth")

    custom_args = parser.parse_args()

    if not custom_args.inference:
        train(custom_args)
    else:
        inference(custom_args)
