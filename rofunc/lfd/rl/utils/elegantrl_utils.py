import isaacgym
import torch, sys

from elegantrl.train.run import train_and_evaluate
from elegantrl.train.config import Arguments, build_env
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.agents.AgentSAC import AgentSAC
import gym.spaces
import numpy as np
from elegantrl.envs.isaac_tasks import isaacgym_task_map
from elegantrl.envs.isaac_tasks.base.vec_task import VecTask
from elegantrl.envs.utils.utils import set_seed
from elegantrl.envs.utils.config_utils import load_task_config, get_max_step_from_config
from pprint import pprint
from typing import Dict, Tuple
from rofunc.lfd.rl.tasks import task_map
from elegantrl.envs.IsaacGym import IsaacVecEnv
from hydra._internal.utils import get_args_parser
from rofunc.config.utils import get_config, omegaconf_to_dict
import argparse


class ElegantRLIsaacGymEnvWrapper:
    def __init__(
            self,
            custom_args,
            task_name: str,
            env=None,
            env_num=-1,
            sim_device_id=0,
            rl_device_id=0,
            headless=True,
            should_print=False,
    ):
        # get config
        sys.argv.append("task={}".format(task_name))
        sys.argv.append("sim_device={}".format(custom_args.sim_device))
        sys.argv.append("rl_device={}".format(custom_args.rl_device))
        sys.argv.append("graphics_device_id={}".format(custom_args.graphics_device_id))
        sys.argv.append("headless={}".format(custom_args.headless))
        args = get_args_parser().parse_args()
        cfg = get_config('./learning/rl', 'config', args=args)
        cfg_dict = omegaconf_to_dict(cfg.task)

        env = task_map[task_name](cfg=cfg_dict,
                                  rl_device=cfg.rl_device,
                                  sim_device=cfg.sim_device,
                                  graphics_device_id=cfg.graphics_device_id,
                                  headless=cfg.headless,
                                  virtual_screen_capture=cfg.capture_video,  # TODO: check
                                  force_render=cfg.force_render)

        is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        # is_discrete = not isinstance(env.action_space, gym.spaces.Box)  # Continuous action space

        state_dimension = env.num_obs
        assert isinstance(state_dimension, int)

        action_dim = getattr(env.action_space, 'n') if is_discrete else env.num_acts
        if not is_discrete:
            assert all(getattr(env.action_space, 'high') == np.ones(action_dim))
            assert all(-getattr(env.action_space, 'low') == np.ones(action_dim))

        target_return = 10 ** 10  # TODO:  plan to make `target_returns` optional

        env_config = cfg_dict["env"]
        max_step = get_max_step_from_config(env_config)

        self.device = torch.device(rl_device_id)
        self.env = env
        self.env_num = env.num_envs
        self.env_name = custom_args.task
        self.max_step = max_step
        self.state_dim = state_dimension
        self.action_dim = action_dim
        self.if_discrete = is_discrete
        self.target_return = target_return

        if should_print:
            pprint(
                {
                    "num_envs": env.num_envs,
                    "env_name": custom_args.task,
                    "max_step": max_step,
                    "state_dim": state_dimension,
                    "action_dim": action_dim,
                    "if_discrete": is_discrete,
                    "target_return": target_return,
                }
            )

    @staticmethod
    def _override_default_env_num(num_envs: int, config_args: Dict):
        """Overrides the default number of environments if it's passed in.

        Args:
            num_envs (int): new number of environments.
            config_args (Dict): configuration retrieved.
        """
        if num_envs > 0:
            config_args["env"]["numEnvs"] = num_envs

    def reset(self) -> torch.Tensor:
        """Resets the environments in the VecTask that need to be reset.

        Returns:
            torch.Tensor: the next states in the simulation.
        """
        observations = self.env.reset()['obs'].to(self.device)
        return observations

    def step(
            self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Steps through the vectorized environment.

        Args:
            actions (torch.Tensor): a multidimensional tensor of actions to perform on
                *each* environment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]: a tuple containing
                observations, rewards, dones, and extra info.
        """
        observations_dict, rewards, dones, info_dict = self.env.step(actions)
        observations = observations_dict["obs"].to(self.device)
        return observations, rewards.to(self.device), dones.to(self.device), info_dict


def demo(custom_args):
    env_name = custom_args.task
    if custom_args.agent.lower() == "ppo":
        agent_class = AgentPPO
    elif custom_args.agent.lower() == "sac":
        agent_class = AgentSAC
    else:
        raise ValueError("Unknown agent")
    env_func = IsaacVecEnv
    gpu_id = 0

    env_args = {
        'env_num': 2048,
        'env_name': env_name,
        'max_step': 1000,
        'state_dim': 41,
        'action_dim': 18,
        'if_discrete': False,
        'target_return': 60000.,

        'sim_device_id': gpu_id,
        'rl_device_id': gpu_id,
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

    train_and_evaluate(args)


if __name__ == '__main__':
    gpu_id = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CURICabinet")
    parser.add_argument("--agent", type=str, default="ppo")
    parser.add_argument("--sim_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--rl_device", type=str, default="cuda:{}".format(gpu_id))
    parser.add_argument("--graphics_device_id", type=int, default=gpu_id)
    parser.add_argument("--headless", type=str, default="False")
    parser.add_argument("--train", action="store_false", help="turn to train mode while adding this argument")
    custom_args = parser.parse_args()

    demo(custom_args)
