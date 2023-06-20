"""
 Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from typing import Union, Tuple, Optional, List

import gym
import gymnasium
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .utils import build_mlp, init_layers, activation_func
from rofunc.config.utils import omegaconf_to_dict


class BaseCritic(nn.Module):
    def __init__(self, cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space, List]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 cfg_name: str = 'critic'):
        super().__init__()
        self.cfg = cfg
        if isinstance(observation_space, List):
            self.state_dim = 0
            for i in range(len(observation_space)):
                if isinstance(observation_space[i], gym.Space) or isinstance(observation_space[i], gymnasium.Space):
                    self.state_dim += observation_space[i].shape[0]
                elif isinstance(observation_space[i], int):
                    self.state_dim += observation_space[i]
                else:
                    raise ValueError(f'observation_space[{i}] is not a valid type.')
        else:
            if isinstance(observation_space, gym.Space) or isinstance(observation_space, gymnasium.Space):
                self.state_dim = observation_space.shape[0]
            else:
                self.state_dim = observation_space
        self.action_dim = action_space.shape[0]
        cfg_dict = omegaconf_to_dict(cfg)
        self.mlp_hidden_dims = cfg_dict[cfg_name]['mlp_hidden_dims']
        self.mlp_activation = activation_func(cfg_dict[cfg_name]['mlp_activation'])

        self.backbone_net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(torch.zeros((self.state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((self.state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std  # todo state_norm

    def value_re_norm(self, value: Tensor) -> Tensor:
        return value * self.value_std + self.value_avg  # todo value_norm

    def freeze_parameters(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze internal parameters
        :param freeze: freeze (True) or unfreeze (False)
        """
        for parameters in self.parameters():
            parameters.requires_grad = not freeze

    def update_parameters(self, model: torch.nn.Module, polyak: float = 1) -> None:
        """
        Update internal parameters by hard or soft (polyak averaging) update
        - Hard update: :math:`\\theta = \\theta_{net}`
        - Soft (polyak averaging) update: :math:`\\theta = (1 - \\rho) \\theta + \\rho \\theta_{net}`
        :param model: Model used to update the internal parameters
        :param polyak: Polyak hyperparameter between 0 and 1 (default: ``1``).
                       A hard update is performed when its value is 1
        """
        with torch.no_grad():
            # hard update
            if polyak == 1:
                for parameters, model_parameters in zip(self.parameters(), model.parameters()):
                    parameters.data.copy_(model_parameters.data)
            # soft update (use in-place operations to avoid creating new parameters)
            else:
                for parameters, model_parameters in zip(self.parameters(), model.parameters()):
                    parameters.data.mul_(1 - polyak)
                    parameters.data.add_(polyak * model_parameters.data)


class Critic(BaseCritic):
    def __init__(self, cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space, List]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 cfg_name: str = 'critic'):
        super().__init__(cfg, observation_space, action_space, cfg_name)
        self.backbone_net = build_mlp(dims=[self.state_dim, *self.mlp_hidden_dims],
                                      hidden_activation=self.mlp_activation)
        self.value_net = nn.Linear(self.mlp_hidden_dims[-1], 1)
        if self.cfg.use_init:
            init_layers(self.backbone_net, gain=1.0)
            init_layers(self.value_net, gain=1.0)

    def forward(self, state: Tensor, action: Tensor = None) -> Tensor:
        if action is not None:
            state = torch.cat((state, action), dim=1)
        value = self.backbone_net(state)
        value = self.value_net(value)
        return value


