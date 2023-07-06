"""
Humanoid (rl_games)
===========================

Humanoid walking, trained by rl_games
"""

import argparse
import isaacgym

from skrl.trainers.torch import SequentialTrainer

from rofunc.learning.pre_trained_models import model_zoo
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


def eval(custom_args, ckpt_path=None):
    beauty_print("Start evaluating")

    env, agent = setup(custom_args, eval_mode=True)

    # load checkpoint (agent)
    if ckpt_path is None:
        ckpt_path = model_zoo(name="CURICabinetPPO_right_arm.pt")
    agent.load(ckpt_path)

    # Configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 1600, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # evaluate the agent
    trainer.eval()


if __name__ == '__main__':
    gpu_id = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Humanoid")
    parser.add_argument("--agent", type=str, default="PPO")
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
