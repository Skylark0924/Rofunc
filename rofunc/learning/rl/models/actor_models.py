from typing import Union, Tuple, Optional

import gym
import gymnasium
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Beta, Normal

from .utils import build_mlp, init_layers, activation_func


class BaseActor(nn.Module):
    def __init__(self, cfg: DictConfig, observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]]):
        super().__init__()
        self.cfg = cfg
        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.mlp_hidden_dims = cfg.actor.mlp_hidden_dims
        self.mlp_activation = activation_func(cfg.actor.mlp_activation)

        self.backbone_net = None  # build_mlp(dims=[state_dim, *dims, action_dim])

        self.state_avg = nn.Parameter(torch.zeros((self.state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((self.state_dim,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std


class ActorPPO_Beta(BaseActor):
    def __init__(self, cfg: DictConfig, observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]]):
        super().__init__(cfg, observation_space, action_space)
        # Build mlp network except the output layer
        self.backbone_net = build_mlp(dims=[self.state_dim, *self.mlp_hidden_dims],
                                      hidden_activation=self.mlp_activation)
        self.alpha_layer = nn.Linear(self.mlp_hidden_dims[-1], self.action_dim)
        self.beta_layer = nn.Linear(self.mlp_hidden_dims[-1], self.action_dim)
        if self.cfg.Model.use_init:
            init_layers(self.backbone_net, gain=1)
            init_layers([self.alpha_layer, self.beta_layer], gain=0.01)

    def forward(self, state: Tensor):
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
    def __init__(self, cfg: DictConfig, observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]]):
        super().__init__(cfg, observation_space, action_space)
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

    def forward(self, state, action=None):
        state = self.backbone_net(state)
        mean_action = self.mean_layer(state)  # [-1,1]

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

    def get_entropy(self):
        return self.dist.entropy()

    # def get_value(self, state):
    #     state = self.backbone_net(state)
    #     value = self.value_layer(state)
    #     return value


class ActorDiscretePPO(BaseActor):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.backbone_net = build_mlp(dims=[state_dim, *dims, action_dim])
        init_with_orthogonal(self.backbone_net[-1], std=0.1)

        self.ActionDist = torch.distributions.Categorical
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        a_prob = self.backbone_net(state)  # action_prob without softmax
        return a_prob.argmax(dim=1)  # get the indices of discrete action

    def get_action(self, state: Tensor) -> (Tensor, Tensor):
        state = self.state_norm(state)
        a_prob = self.soft_max(self.backbone_net(state))
        a_dist = self.ActionDist(a_prob)
        action = a_dist.sample()
        logprob = a_dist.log_prob(action)
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        state = self.state_norm(state)
        a_prob = self.soft_max(self.backbone_net(state))  # action.shape == (batch_size, 1), action.dtype = torch.int
        dist = self.ActionDist(a_prob)
        logprob = dist.log_prob(action.squeeze(1))
        entropy = dist.entropy()
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.long()


class ActorSAC(BaseActor):
    def __init__(self, cfg: DictConfig, observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]]):
        super().__init__(cfg, observation_space, action_space)
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
        state = self.backbone_net(state)
        mean_action = torch.tanh(self.mean_layer(state))  # [-1,1]

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

    def get_action(self, state):
        state = self.state_norm(state)
        s_enc = self.backbone_net_s(state)  # encoded state
        a_avg, a_std_log = self.backbone_net_a(s_enc).chunk(2, dim=1)
        a_std = a_std_log.clamp(-16, 2).exp()

        dist = torch.distributions.normal.Normal(a_avg, a_std)
        return dist.rsample().tanh()  # action (re-parameterize)

    def get_action_logprob(self, state):
        state = self.state_norm(state)
        s_enc = self.backbone_net_s(state)  # encoded state
        a_avg, a_std_log = self.backbone_net_a(s_enc).chunk(2, dim=1)
        a_std = a_std_log.clamp(-16, 2).exp()

        dist = torch.distributions.normal.Normal(a_avg, a_std)
        action = dist.rsample()

        action_tanh = action.tanh()
        logprob = dist.log_prob(a_avg)
        logprob -= (-action_tanh.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        return action_tanh, logprob.sum(1)


class ActorFixSAC(ActorSAC):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(dims=dims, state_dim=state_dim, action_dim=action_dim)
        self.soft_plus = torch.nn.Softplus()

    def get_action_logprob(self, state):
        state = self.state_norm(state)
        s_enc = self.backbone_net_s(state)  # encoded state
        a_avg, a_std_log = self.backbone_net_a(s_enc).chunk(2, dim=1)
        a_std = a_std_log.clamp(-16, 2).exp()

        dist = torch.distributions.normal.Normal(a_avg, a_std)
        action = dist.rsample()

        logprob = dist.log_prob(a_avg)
        logprob -= 2 * (math.log(2) - action - self.soft_plus(action * -2))  # fix logprob using SoftPlus
        return action.tanh(), logprob.sum(1)
