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

import gym
import gymnasium
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from typing import Union, Tuple, Optional, List

from rofunc.config.utils import omegaconf_to_dict
from rofunc.learning.RofuncRL.models.utils import build_mlp, init_layers, activation_func, get_space_dim
from rofunc.learning.RofuncRL.state_encoders.base_encoders import EmptyEncoder


class BaseCritic(nn.Module):
    def __init__(self, cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space, List]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 state_encoder: Optional[nn.Module] = EmptyEncoder(),
                 cfg_name: str = 'critic'):
        super().__init__()
        self.cfg = cfg
        self.action_dim = get_space_dim(action_space)
        cfg_dict = omegaconf_to_dict(cfg)
        self.mlp_hidden_dims = cfg_dict[cfg_name]['mlp_hidden_dims']
        self.mlp_activation = activation_func(cfg_dict[cfg_name]['mlp_activation'])

        # state encoder
        self.state_encoder = state_encoder
        if isinstance(self.state_encoder, EmptyEncoder):
            self.state_dim = get_space_dim(observation_space)
        else:
            self.state_dim = self.state_encoder.output_dim
            if isinstance(observation_space, tuple) or isinstance(observation_space, list):
                self.state_dim += get_space_dim(observation_space[1:])

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
                 state_encoder: Optional[nn.Module] = EmptyEncoder(),
                 cfg_name: str = 'critic'):
        super().__init__(cfg, observation_space, action_space, state_encoder, cfg_name)
        self.backbone_net = build_mlp(dims=[self.state_dim, *self.mlp_hidden_dims],
                                      hidden_activation=self.mlp_activation)
        self.value_net = nn.Linear(self.mlp_hidden_dims[-1], 1)
        if self.cfg.use_init:
            init_layers(self.backbone_net, gain=1.0)
            init_layers(self.value_net, gain=1.0)

    def forward(self, state: Tensor, action: Tensor = None) -> Tensor:
        state = self.state_encoder(state)
        if action is not None:
            state = torch.cat((state, action), dim=1)
        value = self.backbone_net(state)
        value = self.value_net(value)
        return value
