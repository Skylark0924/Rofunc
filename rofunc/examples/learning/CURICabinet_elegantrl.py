"""
CURICabinet (ElegantRL)
===========================

Open drawers with a humanoid CURI robot, trained by ElegantRL
"""

import argparse
import isaacgym

from elegantrl.train.run import train_and_evaluate
from elegantrl.train.config import Arguments, build_env
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.agents.AgentSAC import AgentSAC
from elegantrl.envs.IsaacGym import IsaacVecEnv

from rofunc.utils.logger.beauty_logger import beauty_print
from rofunc.lfd.rl.utils.elegantrl_utils import ElegantRLIsaacGymEnvWrapper
from rofunc.data.models import model_zoo


def setup(custom_args, eval_mode=False):
    if custom_args.agent.lower() == "ppo":
        agent_class = AgentPPO
    elif custom_args.agent.lower() == "sac":
        agent_class = AgentSAC
    else:
        raise ValueError("Agent not supported")
    env_func = IsaacVecEnv

    if eval_mode:
        env_num = 16
    else:
        env_num = 2048

    env_args = {
        'env_num': env_num,
        'env_name': custom_args.task,
        'max_step': 1000,
        'state_dim': 41,
        'action_dim': 18,
        'if_discrete': False,
        'target_return': 60000.,
        'sim_device_id': custom_args.sim_device,
        'rl_device_id': custom_args.rl_device,
    }
    env = build_env(env=ElegantRLIsaacGymEnvWrapper(custom_args, "CURICabinet"), env_func=env_func, env_args=env_args)
    args = Arguments(agent_class, env=env)
    args.if_Isaac = True
    args.if_use_old_traj = True
    args.if_use_gae = True
    args.if_use_per = False

    args.reward_scale = 2 ** -4
    args.horizon_len = 32
    args.batch_size = 16384  # minibatch size
    args.repeat_times = 5
    args.gamma = 0.99
    args.lambda_gae_adv = 0.95
    args.learning_rate = 0.0005

    args.eval_gap = 1e6
    args.target_step = 3e8
    args.learner_gpus = 0
    args.random_seed = 0

    return env, args


def train(custom_args):
    beauty_print("Start training")

    env, args = setup(custom_args)

    # start training
    train_and_evaluate(args)


def eval(custom_args, ckpt_path=None):
    # TODO: add support for eval mode
    beauty_print("Start evaluating")

    env, args = setup(custom_args, eval_mode=True)

    # load checkpoint
    if ckpt_path is None:
        ckpt_path = model_zoo(name="CURICabinetPPO_right_arm.pt")
    agent.load(ckpt_path)

    # evaluate the agent
    trainer.eval()


if __name__ == '__main__':
    gpu_id = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CURICabinet")
    parser.add_argument("--agent", type=str, default="ppo")
    parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
    parser.add_argument("--headless", type=str, default="False")
    parser.add_argument("--test", action="store_true", help="turn to test mode while adding this argument")
    custom_args = parser.parse_args()

    if not custom_args.test:
        train(custom_args)
    else:
        # TODO: add support for eval mode
        folder = 'CURICabinetSAC_22-11-27_18-38-53-296354'
        ckpt_path = "/home/ubuntu/Github/Knowledge-Universe/Robotics/Roadmap-for-robot-science/rofunc/examples/learning/runs/{}/checkpoints/best_agent.pt".format(
            folder)
        eval(custom_args, ckpt_path=ckpt_path)
