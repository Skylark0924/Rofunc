# Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, Tuple, Optional, List

import gym
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Beta, Normal

from rofunc.learning.RofuncRL.models.utils import build_mlp, init_layers, activation_func, get_space_dim
from rofunc.learning.RofuncRL.state_encoders.base_encoders import EmptyEncoder


class BaseActor(nn.Module):
    def __init__(self, cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space, List]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 state_encoder: Optional[nn.Module] = EmptyEncoder()):
        super().__init__()
        self.cfg = cfg
        self.action_dim = get_space_dim(action_space)
        self.mlp_hidden_dims = cfg.actor.mlp_hidden_dims
        self.mlp_activation = activation_func(cfg.actor.mlp_activation)

        # state encoder
        self.state_encoder = state_encoder
        if isinstance(self.state_encoder, EmptyEncoder):
            self.state_dim = get_space_dim(observation_space)
        else:
            self.state_dim = self.state_encoder.output_dim

        self.backbone_net = None  # build_mlp(dims=[state_dim, *dims, action_dim])

        self.state_avg = nn.Parameter(torch.zeros((self.state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((self.state_dim,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std

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


class ActorPPO_Beta(BaseActor):
    def __init__(self, cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 state_encoder: Optional[nn.Module] = EmptyEncoder()):
        super().__init__(cfg, observation_space, action_space, state_encoder)
        # Build mlp network except the output layer
        self.backbone_net = build_mlp(dims=[self.state_dim, *self.mlp_hidden_dims],
                                      hidden_activation=self.mlp_activation)
        self.alpha_layer = nn.Linear(self.mlp_hidden_dims[-1], self.action_dim)
        self.beta_layer = nn.Linear(self.mlp_hidden_dims[-1], self.action_dim)
        if self.cfg.Model.use_init:
            init_layers(self.backbone_net, gain=1)
            init_layers([self.alpha_layer, self.beta_layer], gain=0.01)

    def forward(self, state: Tensor):
        state = self.state_encoder(state)
        state = self.backbone_net(state)
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(state)) + 1.0
        beta = F.softplus(self.beta_layer(state)) + 1.0
        return alpha, beta

    def get_dist(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, state):
        alpha, beta = self.forward(state)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class ActorPPO_Gaussian(BaseActor):
    def __init__(self, cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 state_encoder: Optional[nn.Module] = EmptyEncoder()):
        """
        Gaussian policy network for PPO
        ActorPPO_Gaussian(
              (mlp_activation): ELU(alpha=1.0)
              (backbone_net): Sequential(
                (0): Linear(in_features=self.state_dim, out_features=self.mlp_hidden_dims[0], bias=True)
                (1): hidden_activation
                (2): Linear(in_features=self.mlp_hidden_dims[0], out_features=self.mlp_hidden_dims[1], bias=True)
                (3): hidden_activation
                ...
                (4): Linear(in_features=self.mlp_hidden_dims[-2], out_features=self.mlp_hidden_dims[-1], bias=True)
                (5): hidden_activation
              )
              (mean_layer): Linear(in_features=self.mlp_hidden_dims[-1], out_features=self.action_dim, bias=True)
              (value_layer): Linear(in_features=self.mlp_hidden_dims[-1], out_features=self.action_dim, bias=True)
            )
        :param cfg: model config
        :param observation_space: 
        :param action_space:
        :param state_encoder:
        """
        super().__init__(cfg, observation_space, action_space, state_encoder)
        # Build mlp network except the output layer
        self.backbone_net = build_mlp(dims=[self.state_dim, *self.mlp_hidden_dims],
                                      hidden_activation=self.mlp_activation)
        self.mean_layer = nn.Linear(self.mlp_hidden_dims[-1], self.action_dim)
        # Use 'nn.Parameter' to train log_std automatically
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        self.value_layer = nn.Linear(self.mlp_hidden_dims[-1], 1)
        self.dist = None
        if self.cfg.use_init:
            init_layers(self.backbone_net, gain=1)
            init_layers(self.mean_layer, gain=0.01)

    def forward(self, state, action=None, deterministic=False):
        state = self.state_encoder(state)
        state = self.backbone_net(state)
        if self.cfg.use_action_out_tanh:
            output_action = self.cfg.action_scale * torch.tanh(self.mean_layer(state))  # [-action_scale, action_scale]
        else:
            output_action = self.cfg.action_scale * self.mean_layer(state)

        log_prob = None
        if not deterministic:
            log_std = self.log_std
            if self.cfg.use_log_std_clip:
                log_std = torch.clamp(log_std, self.cfg.log_std_clip_min, self.cfg.log_std_clip_max)

            self.dist = Normal(output_action, log_std.exp())  # Get the Gaussian distribution

            # sample using the re-parameterization trick
            if action is None:
                action = self.dist.rsample()
            if self.cfg.use_action_clip:
                action = torch.clamp(action, -self.cfg.action_clip, self.cfg.action_clip)  # [-max,max]
            log_prob = self.dist.log_prob(action).sum(dim=-1, keepdim=True)
            output_action = action
        return output_action, log_prob

    def get_entropy(self):
        return self.dist.entropy()

    def get_value(self, state):
        state = self.state_encoder(state)
        state = self.backbone_net(state)
        value = self.value_layer(state)
        return value


class ActorSAC(BaseActor):
    def __init__(self, cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 state_encoder: Optional[nn.Module] = EmptyEncoder()):
        super().__init__(cfg, observation_space, action_space, state_encoder)
        # Build mlp network except the output layer
        self.backbone_net = build_mlp(dims=[self.state_dim, *self.mlp_hidden_dims],
                                      hidden_activation=self.mlp_activation)
        self.mean_layer = nn.Linear(self.mlp_hidden_dims[-1], self.action_dim)

        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        self.dist = None
        if self.cfg.use_init:
            init_layers(self.backbone_net, gain=1)
            init_layers(self.mean_layer, gain=0.01)

    def forward(self, state, action=None):
        state = self.state_encoder(state)
        state = self.backbone_net(state)
        mean_action = self.cfg.action_scale * torch.tanh(self.mean_layer(state))  # [-1,1]

        log_std = self.log_std
        if self.cfg.use_log_std_clip:
            log_std = torch.clamp(log_std, self.cfg.log_std_clip_min, self.cfg.log_std_clip_max)

        self.dist = Normal(mean_action, log_std.exp())  # Get the Gaussian distribution

        # sample using the re-parameterization trick
        if action is None:
            action = self.dist.rsample()
        if self.cfg.use_action_clip:
            action = torch.clamp(action, -self.cfg.action_clip, self.cfg.action_clip)  # [-max,max]
        log_prob = self.dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob


class ActorTD3(ActorSAC):
    def __init__(self, cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 state_encoder: Optional[nn.Module] = EmptyEncoder()):
        super().__init__(cfg, observation_space, action_space, state_encoder)

    def forward(self, state):
        state = self.state_encoder(state)
        state = self.backbone_net(state)
        mean_action = torch.tanh(self.mean_layer(state))
        return mean_action, None


class ActorAMP(ActorPPO_Gaussian):
    def __init__(self, cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 state_encoder: Optional[nn.Module] = EmptyEncoder()):
        super().__init__(cfg, observation_space, action_space, state_encoder)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), fill_value=-2.9), requires_grad=False)


if __name__ == '__main__':
    from omegaconf import DictConfig

    cfg = DictConfig({'use_init': True, 'action_scale': 1.0, 'use_log_std_clip': True, 'log_std_clip_min': -20,
                      'log_std_clip_max': 2, 'use_action_clip': True, 'action_clip': 1.0, 'actor':
                          {'type': "Gaussian", 'mlp_hidden_dims': [512, 256, 128], 'mlp_activation': "elu",
                           'use_lstm': False, 'lstm_cell_size': 256, 'lstm_use_prev_action': False, 'max_seq_len': 20}})
    model = ActorPPO_Gaussian(cfg=cfg, observation_space=3, action_space=1)
    print(model)
