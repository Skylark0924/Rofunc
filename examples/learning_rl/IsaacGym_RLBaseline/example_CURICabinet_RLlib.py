"""
CURICabinet (RLlib)
===========================

Open drawers with a humanoid CURI robot, trained by RLlib
"""
# For every ray version, add `import isaacgym` before ray.__init__.py
# If you use ray==0.8.6, you need to change the line 608 in ray.rllib.evaluation.sampler.py
# `atari_metrics = _fetch_atari_metrics(base_env)` to `atari_metrics = None`
# And also add `.cpu()` after line 182 of ray.rllib.agents.ppo.ppo_tf_policy.py
# If you use ray==2.2.0, you need to comment out line 23 in ray.rllib.agents.sac.__init__.py

import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from rofunc.config.utils import get_config, omegaconf_to_dict
from rofunc.utils.logger.beauty_logger import beauty_print
from rofunc.learning.RofuncRL.utils.rllib_utils import RLlibIsaacGymVecEnvWrapper
from rofunc.learning.RofuncRL.tasks import Tasks

from hydra._internal.utils import get_args_parser
from tqdm.auto import tqdm

import ray
# from ray.rllib.algorithms.ppo import PPO
# from ray.rllib.algorithms.sac import SAC
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.sac import SACTrainer


def setup(custom_args, eval_mode=False):
    ray.init()

    sys.argv.append("task={}".format(custom_args.task))
    sys.argv.append("train={}{}RLlib".format(custom_args.task, custom_args.agent.upper()))
    sys.argv.append("sim_device={}".format(custom_args.sim_device))
    sys.argv.append("rl_device={}".format(custom_args.rl_device))
    sys.argv.append("graphics_device_id={}".format(custom_args.graphics_device_id))
    sys.argv.append("headless={}".format(custom_args.headless))
    args = get_args_parser().parse_args()
    beauty_print("Agent: {}{}RLlib".format(custom_args.task, custom_args.agent.upper()), 2)
    cfg = get_config('./learning/rl', 'config', args=args)
    task_cfg_dict = omegaconf_to_dict(cfg.task)
    agent_cfg_dict = omegaconf_to_dict(cfg.train)
    # task_cfg_dict['env']['numEnvs'] = 64

    if eval_mode:
        task_cfg_dict['env']['numEnvs'] = 16

    env_config = {"task_name": custom_args.task,
                  "task_cfg_dict": task_cfg_dict,
                  "cfg": cfg}  # config to pass to env class
    agent_cfg_dict["env"] = RLlibIsaacGymVecEnvWrapper
    agent_cfg_dict["env_config"] = env_config
    # agent_cfg_dict["train_batch_size"] = 4096
    # agent_cfg_dict["framework"] = "tf"
    # agent_cfg_dict["sgd_minibatch_size"] = 4096
    # agent_cfg_dict["rollout_fragment_length"] = 1024

    if custom_args.agent.lower() == "ppo":
        agent = PPOTrainer(config=agent_cfg_dict)
    elif custom_args.agent.lower() == "sac":
        agent_cfg_dict["normalize_actions"] = False
        agent = SACTrainer(config=agent_cfg_dict)
    else:
        raise ValueError("Agent not supported")

    return agent, env_config


def train(custom_args):
    beauty_print("Start training")

    agent, _ = setup(custom_args)

    try:
        with tqdm(range(100000)) as pbar:
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
    env = Tasks().task_map[env_config["task_name"]](cfg=env_config["task_cfg_dict"],
                                                    rl_device=env_config["cfg"].rl_device,
                                                    sim_device=env_config["cfg"].sim_device,
                                                    graphics_device_id=env_config["cfg"].graphics_device_id,
                                                    headless=env_config["cfg"].headless,
                                                    virtual_screen_capture=env_config["cfg"].capture_video,
                                                    # TODO: check
                                                    force_render=env_config["cfg"].force_render)
    agent.restore(ckpt_path)

    done = True

    steps = 0
    while steps < 20000:
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
    parser.add_argument("--agent", type=str, default="sac")
    parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
    parser.add_argument("--headless", type=str, default="True")
    parser.add_argument("--test", action="store_true", help="turn to test mode while adding this argument")
    custom_args = parser.parse_args()

    if not custom_args.test:
        train(custom_args)
    else:
        ckpt_path = "/home/ubuntu/ray_results/PPO_RLlibIsaacGymEnvWrapper_2022-12-16_20-10-43n0xkd71_/checkpoint_21505/checkpoint-21505"

        test(custom_args, ckpt_path=ckpt_path)
