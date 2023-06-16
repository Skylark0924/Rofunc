"""
Gym Tasks (SKRL)
===========================

Gym Tasks RL using SKRL
"""

import argparse
import sys
import isaacgym

import gymnasium as gym
from hydra._internal.utils import get_args_parser
from skrl.trainers.torch import SequentialTrainer

from rofunc.config.utils import get_config
from rofunc.learning.RofuncRL.tasks.utils.env_wrappers import wrap_env
from rofunc.learning.RofuncRL.utils.skrl_utils import setup_agent
from rofunc.learning.utils.utils import set_seed
from rofunc.utils.logger.beauty_logger import beauty_print


def train(custom_args):
    beauty_print("Start training")
    # get config
    sys.argv.append("task={}".format(custom_args.task))
    sys.argv.append("train={}{}SKRL".format(custom_args.task, custom_args.agent.upper()))
    sys.argv.append("headless={}".format(custom_args.headless))
    args = get_args_parser().parse_args()
    cfg = get_config('./learning/rl', 'config', args=args)
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


def eval(custom_args, ckpt_path=None):
    beauty_print("Start evaluating")

    # TODO


if __name__ == '__main__':
    gpu_id = 1
    gym_task_name = 'Pendulum-v1'
    # Classic: ['Acrobot-v1', 'CartPole-v1', 'MountainCarContinuous-v0', 'MountainCar-v0', 'Pendulum-v1']
    # Box2D: ['BipedalWalker-v3', 'CarRacing-v1', 'LunarLander-v2']  `pip install gymnasium[box2d]`
    # MuJoCo: ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'InvertedDoublePendulum-v2',
    #          'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']  `pip install -U mujoco-py`

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Gym_{}".format(gym_task_name))  # Start with 'Gym_'
    parser.add_argument("--agent", type=str, default="a2c")  # Available agents: ppo, sac, td3, a2c
    parser.add_argument("--render_mode", type=str, default=None)  # Available render_mode: None, "human", "rgb_array"
    parser.add_argument("--headless", type=str, default="True")
    parser.add_argument("--test", action="store_true", help="turn to test mode while adding this argument")
    custom_args = parser.parse_args()

    if not custom_args.test:
        train(custom_args)
    else:
        eval(custom_args)
