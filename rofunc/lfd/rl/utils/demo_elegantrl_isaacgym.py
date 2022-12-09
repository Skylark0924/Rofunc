import isaacgym
import torch

from elegantrl.train.run import train_and_evaluate
from elegantrl.train.config import Arguments, build_env
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.agents.AgentSAC import AgentSAC
from elegantrl.envs.IsaacGym import IsaacVecEnv


def demo(task, agent):
    env_name = task
    if agent.lower() == "ppo":
        agent_class = AgentPPO
    elif agent.lower() == "sac":
        agent_class = AgentSAC
    else:
        raise ValueError("Unknown agent")
    env_func = IsaacVecEnv
    gpu_id = 0

    if env_name == 'Ant':
        env_args = {
            'env_num': 2048,
            'env_name': env_name,
            'max_step': 1000,
            'state_dim': 60,
            'action_dim': 8,
            'if_discrete': False,
            'target_return': 6000.,

            'sim_device_id': gpu_id,
            'rl_device_id': gpu_id,
        }
        env = build_env(env_func=env_func, env_args=env_args)
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

    elif env_name == 'Humanoid':
        env_args = {
            'env_num': 1024,
            'env_name': env_name,
            'max_step': 1000,
            'state_dim': 108,
            'action_dim': 21,
            'if_discrete': False,
            'target_return': 15000.,

            'sim_device_id': gpu_id,
            'rl_device_id': gpu_id,
        }
        env = build_env(env_func=env_func, env_args=env_args)
        args = Arguments(agent_class, env=env)
        args.if_Isaac = True
        args.if_use_old_traj = True
        args.if_use_gae = True
        args.if_use_per = True

        args.reward_scale = 0.01
        args.horizon_len = 32
        args.batch_size = 8192
        args.repeat_times = 5
        args.gamma = 0.99
        args.lambda_gae_adv = 0.95
        args.learning_rate = 0.0005

    args.eval_gap = 1e6
    args.target_step = 3e8
    args.learner_gpus = 0
    args.random_seed = 0

    train_and_evaluate(args)


if __name__ == '__main__':
    task = "Ant"
    agent = "PPO"
    demo(task, agent)
