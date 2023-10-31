"""
OpenAIGym Tasks (SKRL)
===========================

OpenAIGym Tasks RL using SKRL
"""

import argparse

import gymnasium as gym
from skrl.trainers.torch import SequentialTrainer

from rofunc.config.utils import get_config
from rofunc.learning.utils.env_wrappers import wrap_env
from rofunc.learning.RofuncRL.utils.skrl_utils import setup_agent
from rofunc.learning.utils.utils import set_seed
from rofunc.utils.logger.beauty_logger import beauty_print


def train(custom_args):
    # get config
    args_overrides = ["task={}".format(custom_args.task),
                      "train={}{}SKRL".format(custom_args.task, custom_args.agent.upper()),
                      "headless={}".format(custom_args.headless)]
    cfg = get_config('./learning/rl', 'config', args=args_overrides)
    gym_task_name = custom_args.task.split('_')[1]
    cfg.task.name = gym_task_name

    set_seed(cfg.seed)

    env = gym.make(gym_task_name, render_mode=custom_args.render_mode)
    if custom_args.agent == 'a2c':
        env = gym.vector.make(gym_task_name, render_mode=custom_args.render_mode, num_envs=10, asynchronous=False)
    env = wrap_env(env, seed=cfg.seed, verbose=False)
    agent = setup_agent(cfg, custom_args, env)

    # Configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 100000, "headless": custom_args.headless}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start training
    trainer.train()


def inference(custom_args, ckpt_path=None):
    beauty_print("Start evaluating")

    # TODO


if __name__ == '__main__':
    gym_task_name = 'Pendulum-v1'
    # Available tasks:
    # Classic: ['Acrobot-v1', 'CartPole-v1', 'MountainCarContinuous-v0', 'MountainCar-v0', 'Pendulum-v1']
    # Box2D: ['BipedalWalker-v3', 'CarRacing-v1', 'LunarLander-v2']  `pip install gymnasium[box2d]`
    # MuJoCo: ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'InvertedDoublePendulum-v2',
    #          'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']  `pip install -U mujoco-py`

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Gym_{}".format(gym_task_name))  # Start with 'Gym_'
    parser.add_argument("--agent", type=str, default="ppo")  # Available agents: ppo, sac, td3, a2c
    parser.add_argument("--render_mode", type=str, default=None)  # Available render_mode: None, "human", "rgb_array"
    parser.add_argument("--headless", type=str, default="True")
    parser.add_argument("--inference", action="store_true", help="turn to test mode while adding this argument")
    parser.add_argument("--ckpt_path", type=str, default=None)
    custom_args = parser.parse_args()

    if not custom_args.inference:
        train(custom_args)
    else:
        inference(custom_args, custom_args.ckpt_path)
