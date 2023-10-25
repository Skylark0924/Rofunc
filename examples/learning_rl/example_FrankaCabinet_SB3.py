"""
FrankaCabinet (Stable Baseline 3)
===========================

Open drawers with a Franka robot, trained by Stable Baseline 3
"""

import argparse
import sys

from hydra._internal.utils import get_args_parser
# from isaacgym_utils.draw import draw_transforms
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from rofunc.config.utils import get_config, omegaconf_to_dict
from rofunc.learning.RofuncRL.utils.stb3_utils import StableBaseline3Wrapper
from rofunc.utils.logger.beauty_logger import beauty_print


# def custom_draws(scene):
#     franka = scene.get_asset('franka')
#     for env_idx in scene.env_idxs:
#         ee_transform = franka.get_ee_transform(env_idx, 'franka')
#         draw_transforms(scene, [env_idx], [ee_transform])


# def learn_cb(local_vars, global_vars):
#     vec_env.render(custom_draws=custom_draws)

def setup(custom_args, eval_mode=False):
    # set the seed for reproducibility
    # set_seed(42)

    # get config
    sys.argv.append("task={}".format("FrankaCabinet"))
    sys.argv.append("sim_device={}".format(custom_args.sim_device))
    sys.argv.append("rl_device={}".format(custom_args.rl_device))
    sys.argv.append("graphics_device_id={}".format(custom_args.graphics_device_id))
    sys.argv.append("train={}".format("FrankaCabinetPPOStb3"))
    args = get_args_parser().parse_args()
    cfg = get_config('./learning/rl', 'config', args=args, debug=True)
    cfg_task_dict = omegaconf_to_dict(cfg.task)

    # if eval_mode:
    cfg_task_dict['env']['numEnvs'] = 16

    env = Tasks().task_map["FrankaCabinet"](cfg=cfg_task_dict,
                                            rl_device=cfg.rl_device,
                                            sim_device=cfg.sim_device,
                                            graphics_device_id=cfg.graphics_device_id,
                                            headless=cfg.headless,
                                            virtual_screen_capture=cfg.capture_video,
                                            force_render=cfg.force_render)
    env = Monitor(StableBaseline3Wrapper(env))

    if eval_mode:
        agent = PPO.load("dqn_lunar", env=env)
    else:
        agent = PPO('MlpPolicy', env=env, verbose=1, tensorboard_log=custom_args.logdir, **cfg.train.ppo)

    return cfg, env, agent


def train(custom_args):
    beauty_print("Start training")

    cfg, env, agent = setup(custom_args)

    agent.learn(total_timesteps=cfg.train.total_timesteps, reset_num_timesteps=False)
    agent.save("FrankaCabinet_stb3")


def eval(custom_args):
    beauty_print("Start evaluation")

    cfg, env, agent = setup(custom_args, eval_mode=True)

    mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes=10)
    print("mean_reward: {}, std_reward: {}".format(mean_reward, std_reward))

    obs = env.reset()
    for i in range(1000):
        action, _states = agent.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', '-l', type=str, default='runs/tb')
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--train", action="store_false", help="turn to train mode while adding this argument")
    custom_args = parser.parse_args()

    if custom_args.train:
        train(custom_args)
    else:
        eval(custom_args)
