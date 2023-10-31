"""
FrankaCabinet (SKRL)
===========================

Open drawers with a Franka robot, trained by SKRL
"""

import argparse
import isaacgym

from skrl.trainers.torch import SequentialTrainer

from rofunc.learning.RofuncRL.utils.skrl_utils import setup
from rofunc.utils.logger.beauty_logger import beauty_print


def train(custom_args):
    beauty_print("Start training")

    env, agent = setup(custom_args)

    # Configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 100000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start training
    trainer.train()


def inference(custom_args, ckpt_path=None):
    beauty_print("Start evaluating")
    custom_args.headless = False

    env, agent = setup(custom_args, eval_mode=True)

    # load checkpoint (agent)
    agent.load(ckpt_path)

    # Configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 1600, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # evaluate the agent
    trainer.eval()


if __name__ == '__main__':
    gpu_id = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="FrankaCabinet")
    parser.add_argument("--agent", type=str, default="PPO")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
    parser.add_argument("--headless", type=str, default="True")
    parser.add_argument("--inference", action="store_true", help="turn to test mode while adding this argument")
    parser.add_argument("--ckpt_path", type=str, default=None)
    custom_args = parser.parse_args()

    if not custom_args.inference:
        train(custom_args)
    else:
        inference(custom_args, custom_args.ckpt_path)
