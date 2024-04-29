"""
OpenAIGym Tasks (RofuncRL)
===========================

OpenAIGym Tasks RL using RofuncRL
"""

import argparse

import gymnasium as gym

from rofunc.config.utils import get_config
from rofunc.learning.RofuncRL.trainers import Trainers
from rofunc.learning.utils.utils import set_seed


def train(custom_args):
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}RofuncRL".format(custom_args.task, custom_args.agent.upper()),
                      "rl_device={}".format(custom_args.rl_device),
                      "headless={}".format(custom_args.headless)]
    cfg = get_config('./learning/rl', 'config', args=args_overrides)
    gym_task_name = custom_args.task.split('_')[1]
    cfg.task.name = gym_task_name

    set_seed(cfg.train.Trainer.seed)

    env = gym.make(gym_task_name, render_mode=custom_args.render_mode)
    if custom_args.agent == 'a2c':
        env = gym.vector.make(gym_task_name, render_mode=custom_args.render_mode, num_envs=10, asynchronous=False)

    # Instantiate the RL trainer
    trainer = Trainers().trainer_map[custom_args.agent](cfg=cfg,
                                                        env=env,
                                                        device=cfg.rl_device,
                                                        env_name=custom_args.task)

    # Start training
    trainer.train()


def inference(custom_args):
    ...
    # TODO


if __name__ == '__main__':
    gpu_id = 0
    gym_task_name = 'Pendulum-v1'
    # Classic: ['Acrobot-v1', 'CartPole-v1', 'MountainCarContinuous-v0', 'MountainCar-v0', 'Pendulum-v1']
    # Box2D: ['BipedalWalker-v3', 'CarRacing-v1', 'LunarLander-v2']  `pip install gymnasium[box2d]`
    # MuJoCo: ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'InvertedDoublePendulum-v2',
    #          'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']  `pip install -U mujoco-py`

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Gym_{}".format(gym_task_name))  # Start with 'Gym_'
    parser.add_argument("--agent", type=str, default="a2c")  # Available agents: ppo, sac, td3, a2c
    parser.add_argument("--render_mode", type=str, default=None)  # Available render_mode: None, "human", "rgb_array"
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
