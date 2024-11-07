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

from abc import abstractmethod
from typing import Tuple, Any
import time
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv


class StableBaseline3Wrapper(VecEnv):
    def __init__(self, env: Any, auto_reset_after_done: bool = True) -> None:
        """
        Isaac Gym environment (preview 3) wrapper for StableBaseline3

        :param env: The IsaacGym environment to wrap
        """
        self.action_space = env.act_space
        self.observation_space = env.obs_space
        self.num_envs = env.num_envs
        self._env = env
        super().__init__(self.num_envs, self.observation_space, self.action_space)

        self._auto_reset_after_done = auto_reset_after_done
        self._has_first_reset = False
        self._step_counts = np.zeros(self.num_envs)
        self._episode_rewards = np.zeros(self.num_envs)
        self._max_episode_steps = 500

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: The state, the reward, the done flag, and the info
        :rtype: tuple of torch.Tensor and any other info
        """
        actions = torch.tensor(actions, dtype=torch.float32, device=self._env.device)
        self._obs_dict, rew_buf, reset_buf, info = self._env.step(actions)

        all_obs = self._obs_dict["obs"].cpu().numpy()
        all_rews = rew_buf.view(-1, ).cpu().numpy()
        all_dones = reset_buf.view(-1, ).cpu().numpy()
        all_infos = [{} for _ in range(self.num_envs)]

        self._step_counts += 1
        self._episode_rewards += all_rews

        # if self._auto_reset_after_done:
        #     done_env_idx = np.where(all_dones)[0]
        #     if len(done_env_idx) > 0:
        #         new_obs = self.reset(done_env_idx)
        #         for env_idx in done_env_idx:
        #             all_infos[env_idx]['terminal_observation'] = all_obs[env_idx]
        #         all_obs = new_obs


        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            all_infos = [{"TimeLimit.truncated": True} for _ in range(self.num_envs)]
            all_dones = np.ones_like(all_dones)
        return all_obs, all_rews, all_dones, all_infos

    def _reset(self, env_idxs):
        env_idxs = torch.tensor(env_idxs)
        self._env.reset_idx(env_idxs)

    def reset(self, env_idxs=False) -> torch.Tensor:
        """Reset the environment

        :return: The state of the environment
        :rtype: torch.Tensor
        """

        if not self._has_first_reset or env_idxs is None:
            env_idxs = list(range(self.num_envs))

        if len(env_idxs) > 0:
            self._reset(env_idxs)
            self._has_first_reset = True

        self._step_counts[env_idxs] = 0
        self._episode_rewards[env_idxs] = 0
        self._elapsed_steps = 0
        obs_dict = self._env.reset()
        return obs_dict["obs"].cpu().numpy()

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        pass

    def close(self) -> None:
        """Close the environment
        """
        pass

    def step_async(self, actions):
        pass

    def step_wait(self):
        pass

    def get_attr(self, attr_name, indices=None):
        pass

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def seed(self, seed):
        pass

    def env_is_wrapped(self):
        pass
