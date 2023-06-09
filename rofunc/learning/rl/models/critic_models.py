from typing import Union, Tuple, Optional

import gym
import gymnasium
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .utils import build_mlp, init_layers, activation_func


class BaseCritic(nn.Module):
    def __init__(self, cfg: DictConfig, observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]]):
        super().__init__()
        self.cfg = cfg
        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.mlp_hidden_dims = cfg.critic.mlp_hidden_dims
        self.mlp_activation = activation_func(cfg.critic.mlp_activation)

        self.backbone_net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(torch.zeros((self.state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((self.state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std  # todo state_norm

    def value_re_norm(self, value: Tensor) -> Tensor:
        return value * self.value_std + self.value_avg  # todo value_norm


class Critic(BaseCritic):
    def __init__(self, cfg: DictConfig, observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]]):
        super().__init__(cfg, observation_space, action_space)
        self.backbone_net = build_mlp(dims=[self.state_dim, *self.mlp_hidden_dims], hidden_activation=self.mlp_activation)
        self.value_net = nn.Linear(self.mlp_hidden_dims[-1], 1)
        if self.cfg.use_init:
            init_layers(self.backbone_net, gain=1.0)
            init_layers(self.value_net, gain=1.0)

    def forward(self, state: Tensor) -> Tensor:
        value = self.backbone_net(state)
        value = self.value_net(value)
        return value


class CriticTwin(BaseCritic):  # shared parameter
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim + action_dim, *dims, 2])

        init_with_orthogonal(self.net[-1], std=0.5)

    def forward(self, state, action):
        state = self.state_norm(state)
        values = self.net(torch.cat((state, action), dim=1))
        values = self.value_re_norm(values)
        return values.mean(dim=1)  # mean Q value

    def get_q_min(self, state, action):
        state = self.state_norm(state)
        values = self.net(torch.cat((state, action), dim=1))
        values = self.value_re_norm(values)
        return torch.min(values, dim=1)[0]  # min Q value

    def get_q1_q2(self, state, action):
        state = self.state_norm(state)
        values = self.net(torch.cat((state, action), dim=1))
        values = self.value_re_norm(values)
        return values[:, 0], values[:, 1]  # two Q values
