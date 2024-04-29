"""
D4RL (RofuncRL)
=======================

D4RL tasks with RofuncRL offline RL algorithms (BC, DTrans, CQL, etc.)
"""

"""
Note::

    Please add following line to .bashrc (according to your own path):
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco210/bin
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
"""

import argparse

import gymnasium as gym

from rofunc.config.utils import omegaconf_to_dict, get_config
from rofunc.learning.RofuncRL.tasks import Tasks
from rofunc.learning.RofuncRL.trainers import Trainers
from rofunc.learning.utils.download_datasets import download_d4rl_dataset
from rofunc.learning.utils.utils import set_seed


def train(custom_args):
    # Config task and trainer parameters for Isaac Gym environments
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}RofuncRL".format(custom_args.task, custom_args.agent.upper()),
                      "sim_device={}".format(custom_args.sim_device),
                      "rl_device={}".format(custom_args.rl_device),
                      "graphics_device_id={}".format(custom_args.graphics_device_id),
                      "headless={}".format(custom_args.headless)]
    cfg = get_config('./learning/rl', 'config', args=args_overrides)

    download_d4rl_dataset(save_dir='../data/D4RL')

    set_seed(cfg.train.Trainer.seed)

    # Instantiate the Isaac Gym environment
    env = gym.make(f'{custom_args.task}-v3')

    # Instantiate the RL trainer
    trainer = Trainers().trainer_map[custom_args.agent](cfg=cfg,
                                                        env=env,
                                                        device=cfg.rl_device,
                                                        env_name=custom_args.task)

    # Start training
    trainer.train()


def inference(custom_args):
    # Config task and trainer parameters for Isaac Gym environments
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}RofuncRL".format(custom_args.task, custom_args.agent.upper()),
                      "sim_device={}".format(custom_args.sim_device),
                      "rl_device={}".format(custom_args.rl_device),
                      "graphics_device_id={}".format(custom_args.graphics_device_id),
                      "headless={}".format(False),
                      "num_envs={}".format(16)]
    cfg = get_config('./learning/rl', 'config', args=args_overrides)
    cfg_dict = omegaconf_to_dict(cfg.task)

    set_seed(cfg.train.Trainer.seed)

    # Instantiate the Isaac Gym environment
    infer_env = Tasks().task_map[custom_args.task](cfg=cfg_dict,
                                                   rl_device=cfg.rl_device,
                                                   sim_device=cfg.sim_device,
                                                   graphics_device_id=cfg.graphics_device_id,
                                                   headless=cfg.headless,
                                                   virtual_screen_capture=cfg.capture_video,  # TODO: check
                                                   force_render=cfg.force_render)

    # Instantiate the RL trainer
    trainer = Trainers().trainer_map[custom_args.agent](cfg=cfg,
                                                        env=infer_env,
                                                        device=cfg.rl_device,
                                                        env_name=custom_args.task)
    # load checkpoint
    if custom_args.ckpt_path is None:
        raise ValueError("Please specify the checkpoint path for inference.")
    trainer.agent.load_ckpt(custom_args.ckpt_path)

    # Start inference
    trainer.inference()


if __name__ == '__main__':
    gpu_id = 0

    parser = argparse.ArgumentParser()
    # Available tasks: Hopper, HalfCheetah, Walker2d, Reacher2d
    parser.add_argument("--task", type=str, default="Hopper")
    parser.add_argument("--agent", type=str, default="dtrans")  # dtrans
    parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
    parser.add_argument("--headless", type=str, default="True")
    parser.add_argument("--inference", action="store_true", help="turn to inference mode while adding this argument")
    parser.add_argument("--ckpt_path", type=str, default=None)
    custom_args = parser.parse_args()

    if not custom_args.inference:
        train(custom_args)
    else:
        inference(custom_args)
