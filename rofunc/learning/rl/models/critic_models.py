import torch
from torch import Tensor
from omegaconf import DictConfig

from .base_model import BaseCritic, build_mlp, init_with_orthogonal, activation_func


class Critic(BaseCritic):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim + action_dim, *dims, 1])

        init_with_orthogonal(self.net[-1], std=0.5)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        state = self.state_norm(state)
        values = self.net(torch.cat((state, action), dim=1))
        values = self.value_re_norm(values)
        return values.squeeze(dim=1)  # q value


class CriticPPO(BaseCritic):
    """Value Network for PPO"""
    def __init__(self, cfg: DictConfig, observation_space: int, action_space: int):
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.mlp_hidden_dims = cfg.mlp_hidden_dims
        self.mlp_activation = activation_func(cfg.mlp_activation)

        self.net = build_mlp(dims=[state_dim, *self.mlp_hidden_dims, 1], hidden_activation=self.mlp_activation)

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        value = self.net(state)
        value = self.value_re_norm(value)
        return value.squeeze(1).reshape((-1, 1))  # q value


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
