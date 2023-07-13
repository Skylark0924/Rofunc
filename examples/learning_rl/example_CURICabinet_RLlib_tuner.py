"""
CURICabinet (RLlib Tuner)
===========================

Open drawers with a humanoid CURI robot, trained by RLlib Tuner
"""

import isaacgym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray import air
from ray import tune
from rofunc.learning.RofuncRL.utils.rllib_utils import RLlibIsaacGymVecEnvWrapper
from hydra._internal.utils import get_args_parser
import sys, argparse
from rofunc.config.utils import get_config, omegaconf_to_dict
from ray.tune.logger import pretty_print

gpu_id = 0
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="CURICabinet")
parser.add_argument("--agent", type=str, default="ppo")
parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
parser.add_argument("--headless", type=str, default="True")
parser.add_argument("--test", action="store_true", help="turn to test mode while adding this argument")
custom_args = parser.parse_args()

sys.argv.append("task={}".format(custom_args.task))
# sys.argv.append("num_envs={}".format(1))
sys.argv.append("train={}{}RLlib".format(custom_args.task, custom_args.agent.upper()))
sys.argv.append("sim_device={}".format(custom_args.sim_device))
sys.argv.append("rl_device={}".format(custom_args.rl_device))
sys.argv.append("graphics_device_id={}".format(custom_args.graphics_device_id))
sys.argv.append("headless={}".format(custom_args.headless))
args = get_args_parser().parse_args()
cfg = get_config('./learning/rl', 'config', args=args)
task_cfg_dict = omegaconf_to_dict(cfg.task)

env_config = {"task_name": custom_args.task,
              "task_cfg_dict": task_cfg_dict,
              "cfg": cfg}  # config to pass to env class

# config = PPOConfig()
# config = config.rollouts(num_rollout_workers=0)
# config = config.resources(num_gpus=1)
# # Print out some default values.
# print(config.clip_param)
# # Update the config object.
# config.training(
#     lr=0.0001, clip_param=0.2
# )
# # Set the config object's env.
# config = config.environment(env=RLlibIsaacGymVecEnvWrapper, env_config=env_config)
# # Use to_dict() to get the old-style python config dict
# # when running with tune.
# tune.Tuner(
#     "PPO",
#     run_config=air.RunConfig(stop={"episode_reward_mean": 500}),
#     param_space=config.to_dict(),
# ).fit()
algo = (
    SACConfig()
    .training(train_batch_size=1024)
    .rollouts(num_rollout_workers=0)
    .resources(num_gpus=1)
    .environment(env=RLlibIsaacGymVecEnvWrapper, env_config=env_config)
    .build()
)

for i in range(100):
    result = algo.train()
    # print(pretty_print(result))
    print(result["episode_reward_mean"])

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")