from typing import Union, Tuple, Dict, Optional

import gym
import gymnasium
import numpy as np
import torch

from rofunc.learning.rl.models.base_model import BaseModel
from rofunc.learning.rl.utils.memory import Memory


class BaseAgent:
    def __init__(self,
                 cfg: Optional[dict],
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 models: Dict[str, BaseModel],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 ):
        """
        Base class of Rofunc RL Agents.
        :param cfg: Custom configuration
        :param observation_space: Observation/state space or shape
        :param action_space: Action space or shape
        :param memory: Memory for storing transitions
        :param device: Device on which the torch tensor is allocated
        """
        self.cfg = cfg
        self.observation_space = observation_space
        self.action_space = action_space
        self.models = models
        self.memory = memory
        self.device = device

        '''Checkpoint'''
        self.checkpoint_modules = {}
        self.checkpoint_interval = self.cfg.get("experiment", {}).get("checkpoint_interval", 1000)
        self.checkpoint_store_separately = self.cfg.get("experiment", {}).get("store_separately", False)
        self.checkpoint_best_modules = {"timestep": 0, "reward": -2 ** 31, "saved": False, "modules": {}}

        '''Logging'''
        self.cumulative_rewards = None
        self.cumulative_timesteps = None

    def act(self, states: torch.Tensor):
        """
        Make a decision based on the current state.
        :param states: current environment states
        :return: action
        """
        actions = self.actor_model.get_action(states)
        return actions

    def transition_record(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor,
                          rewards: torch.Tensor, dones: torch.Tensor, infos: torch.Tensor):
        """
        Record the transition.
        """
        if self.cumulative_rewards is None:
            self.cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
            self.cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)

        self.cumulative_rewards.add_(rewards)
        self.cumulative_timesteps.add_(1)

        # check ended episodes
        finished_episodes = (terminated + truncated).nonzero(as_tuple=False)
        if finished_episodes.numel():
            # storage cumulative rewards and timesteps
            self._track_rewards.extend(self._cumulative_rewards[finished_episodes][:, 0].reshape(-1).tolist())
            self._track_timesteps.extend(self._cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())

            # reset the cumulative rewards and timesteps
            self._cumulative_rewards[finished_episodes] = 0
            self._cumulative_timesteps[finished_episodes] = 0

        # record data
        if self.write_interval > 0:
            self.tracking_data["Reward / Instantaneous reward (max)"].append(torch.max(rewards).item())
            self.tracking_data["Reward / Instantaneous reward (min)"].append(torch.min(rewards).item())
            self.tracking_data["Reward / Instantaneous reward (mean)"].append(torch.mean(rewards).item())

            if len(self._track_rewards):
                track_rewards = np.array(self._track_rewards)
                track_timesteps = np.array(self._track_timesteps)

                self.tracking_data["Reward / Total reward (max)"].append(np.max(track_rewards))
                self.tracking_data["Reward / Total reward (min)"].append(np.min(track_rewards))
                self.tracking_data["Reward / Total reward (mean)"].append(np.mean(track_rewards))

                self.tracking_data["Episode / Total timesteps (max)"].append(np.max(track_timesteps))
                self.tracking_data["Episode / Total timesteps (min)"].append(np.min(track_timesteps))
                self.tracking_data["Episode / Total timesteps (mean)"].append(np.mean(track_timesteps))

    def update(self):
        """
        Update the agent model parameters.
        """
        raise NotImplementedError

    def save_ckpt(self, path: str):
        """
        Save the agent model parameters to a checkpoint.
        :param path:
        :return:
        """
        modules = {}
        for name, module in self.checkpoint_modules.items():
            modules[name] = self._get_internal_value(module)
        torch.save(modules, path)

    def load_ckpt(self, path: str):
        """
        Load the agent model parameters from a checkpoint.
        :param path:
        :return:
        """
        modules = torch.load(path, map_location=self.device)
        if type(modules) is dict:
            for name, data in modules.items():
                module = self.checkpoint_modules.get(name, None)
                if module is not None:
                    if hasattr(module, "load_state_dict"):
                        module.load_state_dict(data)
                        if hasattr(module, "eval"):
                            module.eval()
                    else:
                        raise NotImplementedError
                else:
                    self.rofunc_logger.warning(
                        "Cannot load the {} module. The agent doesn't have such an instance".format(name))
