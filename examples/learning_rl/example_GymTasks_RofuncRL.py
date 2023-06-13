"""
Gym Tasks (RofuncRL)
===========================

Gym Tasks RL using RofuncRL
"""

import argparse
import sys
import gymnasium as gym
import isaacgym

from hydra._internal.utils import get_args_parser

from rofunc.config.utils import get_config
from rofunc.learning.rl.trainers import trainer_map
from rofunc.learning.utils.utils import set_seed
from rofunc.utils.logger.beauty_logger import beauty_print


def train(custom_args):
    beauty_print("Start training")
    # get config
    sys.argv.append("task={}".format(custom_args.task))
    sys.argv.append("train={}{}RofuncRL".format(custom_args.task, custom_args.agent.upper()))
    sys.argv.append("sim_device={}".format(custom_args.sim_device))
    sys.argv.append("rl_device={}".format(custom_args.rl_device))
    sys.argv.append("graphics_device_id={}".format(custom_args.graphics_device_id))
    sys.argv.append("headless={}".format(custom_args.headless))
    args = get_args_parser().parse_args()
    cfg = get_config('./learning/rl', 'config', args=args)
    gym_task_name = custom_args.task.split('_')[1]
    cfg.task.name = gym_task_name

    set_seed(cfg.train.Trainer.seed)

    env = gym.make(gym_task_name, render_mode=custom_args.render_mode)

    # Instantiate the RL trainer
    trainer = trainer_map[custom_args.agent](cfg=cfg.train,
                                             env=env,
                                             device=cfg.rl_device)

    # start training
    trainer.train()


def eval(custom_args, ckpt_path=None):
    beauty_print("Start evaluating")

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
    parser.add_argument("--agent", type=str, default="td3")  # Available agents: ppo, sac, td3
    parser.add_argument("--render_mode", type=str, default=None)  # Available render_mode: None, "human", "rgb_array"
    parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
    parser.add_argument("--headless", type=str, default="True")
    parser.add_argument("--test", action="store_true", help="turn to test mode while adding this argument")
    custom_args = parser.parse_args()

    if not custom_args.test:
        train(custom_args)
    else:
        eval(custom_args)
