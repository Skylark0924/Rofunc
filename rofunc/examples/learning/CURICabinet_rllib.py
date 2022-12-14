"""
CURICabinet (RLlib)
===========================

Open drawers with a humanoid CURI robot, trained by RLlib
"""

import argparse
import sys

import isaacgym

from rofunc.config.utils import get_config, omegaconf_to_dict
from rofunc.data.models import model_zoo
from rofunc.utils.logger.beauty_logger import beauty_print
from rofunc.lfd.rl.utils.rllib_utils import RLlibIsaacGymEnvWrapper

from hydra._internal.utils import get_args_parser
from tqdm.auto import tqdm

import ray
from ray.rllib.agents import ppo


def setup(custom_args, eval_mode=False):
    ray.init()

    sys.argv.append("task={}".format(custom_args.task))
    sys.argv.append("sim_device={}".format(custom_args.sim_device))
    sys.argv.append("rl_device={}".format(custom_args.rl_device))
    sys.argv.append("graphics_device_id={}".format(custom_args.graphics_device_id))
    sys.argv.append("headless={}".format(custom_args.headless))
    args = get_args_parser().parse_args()
    cfg = get_config('./learning/rl', 'config', args=args)
    task_cfg_dict = omegaconf_to_dict(cfg.task)

    if eval_mode:
        task_cfg_dict['env']['numEnvs'] = 16

    env_config = {"task_name": custom_args.task,
                  "task_cfg_dict": task_cfg_dict,
                  "cfg": cfg}  # config to pass to env class

    if custom_args.agent.lower() == "ppo":
        agent = ppo.PPOTrainer(env=RLlibIsaacGymEnvWrapper, config={
            "env_config": env_config,
            # "framework": "torch",
            "num_workers": 0,
            # 'explore': True,
            # 'exploration_config': {
            #     'type': 'StochasticSampling'
            #     # 'type': 'Curiosity',
            #     # 'eta': 1.0,
            #     # 'lr': 0.001,
            #     # 'feature_dim': 288,
            #     # "feature_net_config": {
            #     #    "fcnet_hiddens": [],
            #     #    "fcnet_activation": "relu",
            #     # },
            #     # "inverse_net_hiddens": [256],
            #     # "inverse_net_activation": "relu",
            #     # "forward_net_hiddens": [256],
            #     # "forward_net_activation": "relu",
            #     # "beta": 0.2,
            #     # "sub_exploration": {
            #     #    "type": "StochasticSampling",
            #     # }
            # },
            # "num_envs_per_worker": 1,
            'gamma': 0.998,

            'train_batch_size': 2048,
            'sgd_minibatch_size': 2048,
            'rollout_fragment_length': 64,
            'num_sgd_iter': 3,
            'lr': 5e-5,

            'vf_loss_coeff': 0.5,
            'vf_share_layers': True,
            'kl_coeff': 0.0,
            'kl_target': 0.1,
            'clip_param': 0.1,
            'entropy_coeff': 0.005,

            'grad_clip': 1.0,
            'lambda': 0.8,
        })
    else:
        raise ValueError("Agent not supported")

    return agent, env_config


def train(custom_args):
    beauty_print("Start training")

    agent, _ = setup(custom_args)

    try:
        with tqdm(range(32768)) as pbar:
            for i in pbar:
                results = agent.train()
                if i % 64 == 0:
                    avg_reward = results['episode_reward_mean']
                    pbar.set_description(
                        F'Iter: {i}; avg.rew={avg_reward:02f}')
                if i % 1024 == 0:
                    ckpt = agent.save()
                    print(F'saved ckpt = {ckpt}')
    finally:
        ckpt = agent.save()
        print(F'saved ckpt = {ckpt}')


def test(custom_args, ckpt_path):
    beauty_print("Start testing")

    agent, env_config = setup(custom_args, eval_mode=True)
    agent.restore(ckpt_path)
    env = RLlibIsaacGymEnvWrapper(env_config)

    done: bool = True

    steps = 0
    while True:
        if done:
            obs = env.reset()
        action = agent.compute_single_action(obs, explore=False)  # TODO
        obs, reward, done, info = env.step(action)

        steps += 1
        if steps % 1000 == 0:
            print(steps)


if __name__ == '__main__':
    gpu_id = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CURICabinet")
    parser.add_argument("--agent", type=str, default="ppo")
    parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
    parser.add_argument("--headless", type=str, default="False")
    parser.add_argument("--test", action="store_false", help="turn to test mode while adding this argument")
    custom_args = parser.parse_args()

    if not custom_args.test:
        train(custom_args)
    else:
        ckpt_path = "/home/ubuntu/ray_results/PPO_RLlibIsaacGymEnvWrapper_2022-12-14_17-28-420ve1a92i/checkpoint_32768/checkpoint-32768"

        test(custom_args, ckpt_path=ckpt_path)
