"""
FrankaCabinet
===========================

Open a cabinet with a single Franka arm
"""

import argparse
import isaacgym

from skrl.trainers.torch import SequentialTrainer

from rofunc.data.models import model_zoo
from rofunc.examples.learning.base_skrl import setup
from rofunc.utils.logger.beauty_logger import beauty_print


def train(custom_args, task_name):
    beauty_print("Start training")

    env, agent = setup(custom_args, task_name)

    # Configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 24000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start training
    trainer.train()


def eval(custom_args, task_name, ckpt_path=None):
    beauty_print("Start evaluating")

    env, agent = setup(custom_args, task_name, eval_mode=True)

    # load checkpoint
    if ckpt_path is None:
        ckpt_path = model_zoo(name="FrankaCabinetPPO.pt")
    agent.load(ckpt_path)

    # Configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 1600, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # evaluate the agent
    trainer.eval()


if __name__ == '__main__':
    gpu_id = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="td3")
    parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
    parser.add_argument("--headless", type=str, default="False")
    parser.add_argument("--train", action="store_true", help="turn to train mode while adding this argument")
    custom_args = parser.parse_args()

    task_name = "FrankaCabinet"

    if custom_args.train:
        train(custom_args, task_name)
    else:
        eval(custom_args, task_name)
