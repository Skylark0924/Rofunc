"""
FrankaCabinet
===========================

Open a cabinet with a single Franka arm
"""

import argparse
import sys

import isaacgym
import torch

from rofunc.config.utils import get_config, omegaconf_to_dict, dict_to_omegaconf
from rofunc.examples.learning.base_skrl import set_cfg_ppo, set_models_ppo
from rofunc.lfd.rl.tasks import task_map
from rofunc.data.models import model_zoo
from rofunc.utils.logger.beauty_logger import beauty_print
from rofunc.lfd.utils.utils import set_seed

from hydra._internal.utils import get_args_parser

import ray, gym
from ray.rllib.algorithms import ppo
from ray.tune.logger import pretty_print


class EnvWrapperRLlib(gym.Env):
    def __init__(self, env_config):
        env_config = argparse.Namespace(**env_config)
        # env_config = dict_to_omegaconf(env_config)

        # get config
        sys.argv.append("task={}".format("FrankaCabinet"))
        sys.argv.append("sim_device={}".format(env_config.sim_device))
        sys.argv.append("rl_device={}".format(env_config.rl_device))
        sys.argv.append("graphics_device_id={}".format(env_config.graphics_device_id))
        args = get_args_parser().parse_args()
        cfg = get_config('./learning/rl', 'config', args=args)
        cfg_dict = omegaconf_to_dict(cfg.task)

        self.isaac_gym_env = task_map["FrankaCabinet"](cfg=cfg_dict,
                                        rl_device=cfg.rl_device,
                                        sim_device=cfg.sim_device,
                                        graphics_device_id=cfg.graphics_device_id,
                                        headless=cfg.headless,
                                        virtual_screen_capture=cfg.capture_video,  # TODO: check
                                        force_render=cfg.force_render)

        self.action_space = self.isaac_gym_env.act_space
        self.observation_space = self.isaac_gym_env.obs_space

    def reset(self):
        return self.isaac_gym_env.reset()

    def step(self, action):
        return self.isaac_gym_env.step(action)


def setup(custom_args, eval_mode=False):
    # set the seed for reproducibility
    set_seed(42)

    # # get config
    # sys.argv.append("task={}".format("FrankaCabinet"))
    # sys.argv.append("sim_device={}".format(custom_args.sim_device))
    # sys.argv.append("rl_device={}".format(custom_args.rl_device))
    # sys.argv.append("graphics_device_id={}".format(custom_args.graphics_device_id))
    # args = get_args_parser().parse_args()
    # cfg = get_config('./learning/rl', 'config', args=args)
    # cfg_dict = omegaconf_to_dict(cfg.task)
    #
    # if eval_mode:
    #     cfg_dict['env']['numEnvs'] = 16
    #
    # env = task_map["FrankaCabinet"](cfg=cfg_dict,
    #                                 rl_device=cfg.rl_device,
    #                                 sim_device=cfg.sim_device,
    #                                 graphics_device_id=cfg.graphics_device_id,
    #                                 headless=cfg.headless,
    #                                 virtual_screen_capture=cfg.capture_video,  # TODO: check
    #                                 force_render=cfg.force_render)

    # env = EnvWrapperRLlib(env, None)

    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config["env_config"] = vars(custom_args)
    config["num_workers"] = 1
    config["framework"] = 'torch'
    agent = ppo.PPO(env=EnvWrapperRLlib, config=config)

    return agent


def train(custom_args):
    beauty_print("Start training")

    agent = setup(custom_args)

    for i in range(1000):
        # Perform one iteration of training the policy with PPO
        result = agent.train()
        print(pretty_print(result))

        if i % 100 == 0:
            checkpoint = agent.save()
            print("checkpoint saved at", checkpoint)


# def eval(custom_args, ckpt_path=None):
#     beauty_print("Start evaluating")
#
#     env, agent = setup(custom_args, eval_mode=True)
#
#     # load checkpoint (agent)
#     if ckpt_path is None:
#         ckpt_path = model_zoo(name="FrankaCabinetPPO.pt")
#     agent.load(ckpt_path)
#
#     # Configure and instantiate the RL trainer
#     cfg_trainer = {"timesteps": 1600, "headless": True}
#     trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
#
#     # evaluate the agent
#     trainer.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--train", action="store_false", help="turn to train mode while adding this argument")
    custom_args = parser.parse_args()

    if custom_args.train:
        train(custom_args)
    else:
        eval(custom_args)
