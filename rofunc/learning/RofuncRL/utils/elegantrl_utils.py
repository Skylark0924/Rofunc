#  Copyright (C) 2024, Junjia Liu
# 
#  This file is part of Rofunc.
# 
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
# 
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
# 
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com

import isaacgym
import torch
import sys

import gym.spaces
import numpy as np
from pprint import pprint
from typing import Dict, Tuple

from hydra._internal.utils import get_args_parser

from elegantrl.train.config import Arguments
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.agents.AgentSAC import AgentSAC
from elegantrl.agents.AgentTD3 import AgentTD3
from elegantrl.train.run import init_agent

from rofunc.config.utils import get_config, omegaconf_to_dict
from rofunc.utils.logger.beauty_logger import beauty_print


class ElegantRLIsaacGymEnvWrapper:
    def __init__(self, env, cfg, should_print=False):
        is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        # is_discrete = not isinstance(env.action_space, gym.spaces.Box)  # Continuous action space

        state_dimension = env.num_obs
        assert isinstance(state_dimension, int)

        action_dim = getattr(env.action_space, 'n') if is_discrete else env.num_acts
        if not is_discrete:
            assert all(getattr(env.action_space, 'high') == np.ones(action_dim))
            assert all(-getattr(env.action_space, 'low') == np.ones(action_dim))

        target_return = 10 ** 10  # TODO:  plan to make `target_returns` optional

        self.device = torch.device(cfg.graphics_device_id)
        self.env = env
        self.env_num = env.num_envs
        self.env_name = cfg.task_name
        self.max_step = cfg.task.env.episodeLength
        self.state_dim = state_dimension
        self.action_dim = action_dim
        self.if_discrete = is_discrete
        self.target_return = target_return

        if should_print:
            pprint(
                {
                    "num_envs": env.num_envs,
                    "env_name": cfg.task_name,
                    "max_step": cfg.task.env.episodeLength,
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


def setup(custom_args, eval_mode=False):
    # get config
    sys.argv.append("task={}".format(custom_args.task))
    beauty_print("Agent: {}{}ElegantRL".format(custom_args.task, custom_args.agent.upper()), type="info")
    sys.argv.append("sim_device={}".format(custom_args.sim_device))
    sys.argv.append("rl_device={}".format(custom_args.rl_device))
    sys.argv.append("graphics_device_id={}".format(custom_args.graphics_device_id))
    sys.argv.append("headless={}".format(custom_args.headless))
    args = get_args_parser().parse_args()
    cfg = get_config('./learning/rl', 'config', args=args)
    task_cfg_dict = omegaconf_to_dict(cfg.task)

    if eval_mode:
        task_cfg_dict['env']['numEnvs'] = 16
        cfg.headless = False

    from rofunc.learning.RofuncRL.tasks import task_map
    env = task_map[custom_args.task](cfg=task_cfg_dict,
                                     rl_device=cfg.rl_device,
                                     sim_device=cfg.sim_device,
                                     graphics_device_id=cfg.graphics_device_id,
                                     headless=cfg.headless,
                                     virtual_screen_capture=cfg.capture_video,  # TODO: check
                                     force_render=cfg.force_render)

    env = ElegantRLIsaacGymEnvWrapper(env, cfg)

    if custom_args.agent.lower() == "ppo":
        agent_class = AgentPPO
    elif custom_args.agent.lower() == "sac":
        agent_class = AgentSAC
    elif custom_args.agent.lower() == "td3":
        agent_class = AgentTD3
    else:
        raise ValueError("Agent not supported")

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
    args.learner_gpus = cfg.graphics_device_id
    args.random_seed = 42

    if eval_mode:
        agent = init_agent(args, args.learner_gpus, env)
        return env, agent

    return env, args
