from typing import Union, Tuple, Optional, List

import gym
import gymnasium
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .utils import build_mlp, init_layers, activation_func


class BaseCritic(nn.Module):
    def __init__(self, cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space, List]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]]):
        super().__init__()
        self.cfg = cfg
        if isinstance(observation_space, List):
            self.state_dim = 0
            for i in range(len(observation_space)):
                self.state_dim += observation_space[i].shape[0]
        else:
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
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]]):
        super().__init__(cfg, observation_space, action_space)
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
