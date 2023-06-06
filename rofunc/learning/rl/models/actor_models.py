import math
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Beta, Normal

from .base_model import BaseActor, build_mlp, init_layers, activation_func


class Actor(BaseActor):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: [int], hidden_activation=nn.ReLU,
                 output_activation=None):
        super(Actor, self).__init__(state_dim, action_dim)
        self.net = build_mlp(dims=[state_dim, *hidden_dims, action_dim], hidden_activation=hidden_activation,
                             output_activation=output_activation)
        init_with_orthogonal(self.net[-1], std=0.01)

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state: Tensor) -> Tensor:  # for exploration
        state = self.state_norm(state)
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * self.explore_noise_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)

    def act_noise(self, state: Tensor, action_std: float) -> Tensor:
        state = self.state_norm(state)
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class ActorPPO_Beta(nn.Module):
    def __init__(self, cfg: DictConfig, observation_space: int, action_space: int):
        super().__init__()
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        self.mlp_hidden_dims = cfg.mlp_hidden_dims
        self.mlp_activation = activation_func(cfg.mlp_activation)

        # Build mlp network except the output layer
        self.net = build_mlp(dims=[state_dim, *self.mlp_hidden_dims], hidden_activation=self.mlp_activation)
        self.alpha_layer = nn.Linear(self.mlp_hidden_dims[-1], action_dim)
        self.beta_layer = nn.Linear(self.mlp_hidden_dims[-1], action_dim)
        init_layers(self.net, gain=1)
        init_layers([self.alpha_layer, self.beta_layer], gain=0.01)

    def forward(self, state: Tensor):
        state = self.net(state)
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(state)) + 1.0
        beta = F.softplus(self.beta_layer(state)) + 1.0
        return alpha, beta

    # def act(self, state: Tensor, deterministic=False) -> (Tensor, Tensor):  # for exploration
    #     state = self.state_norm(state)
    #     action_avg = self.net(state)
    #     action_std = self.action_std_log.exp()
    #
    #     dist = self.ActionDist(action_avg, action_std)
    #
    #     if deterministic:
    #         action = dist.mode
    #     else:
    #         action = dist.sample()
    #     log_prob = dist.log_prob(action).sum(1)
    #     return action, log_prob
    #
    # def get_log_prob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
    #     state = self.state_norm(state)
    #     action_avg = self.net(state)
    #     action_std = self.action_std_log.exp()
    #
    #     try:
    #         dist = self.ActionDist(action_avg, action_std)
    #     except:
    #         raise ValueError
    #     log_prob = dist.log_prob(action).sum(1)
    #     entropy = dist.entropy().sum(1)
    #     return log_prob, entropy

    def get_dist(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, state):
        alpha, beta = self.forward(state)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class ActorPPO_Gaussian(nn.Module):
    def __init__(self, cfg: DictConfig, observation_space: int, action_space: int):
        super().__init__()
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        self.mlp_hidden_dims = cfg.mlp_hidden_dims
        self.mlp_activation = activation_func(cfg.mlp_activation)

        # Build mlp network except the output layer
        self.net = build_mlp(dims=[state_dim, *self.mlp_hidden_dims], hidden_activation=self.mlp_activation)
        self.mean_layer = nn.Linear(self.mlp_hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # We use 'nn.Parameter' to train log_std automatically
        init_layers(self.net, gain=1)
        init_layers(self.mean_layer, gain=0.01)

    def forward(self, state):
        if torch.isnan(state).any():
            raise ValueError
        state = self.net(state)
        if torch.isnan(state).any():
            raise ValueError
        mean = torch.tanh(self.mean_layer(state))  # [-1,1]
        return mean

    def get_dist(self, state):
        mean = self.forward(state)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        try:
            dist = Normal(mean, std)  # Get the Gaussian distribution
        except:
            raise ValueError
        return dist


class ActorDiscretePPO(BaseActor):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        init_with_orthogonal(self.net[-1], std=0.1)

        self.ActionDist = torch.distributions.Categorical
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        a_prob = self.net(state)  # action_prob without softmax
        return a_prob.argmax(dim=1)  # get the indices of discrete action

    def get_action(self, state: Tensor) -> (Tensor, Tensor):
        state = self.state_norm(state)
        a_prob = self.soft_max(self.net(state))
        a_dist = self.ActionDist(a_prob)
        action = a_dist.sample()
        logprob = a_dist.log_prob(action)
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        state = self.state_norm(state)
        a_prob = self.soft_max(self.net(state))  # action.shape == (batch_size, 1), action.dtype = torch.int
        dist = self.ActionDist(a_prob)
        logprob = dist.log_prob(action.squeeze(1))
        entropy = dist.entropy()
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.long()


class ActorSAC(BaseActor):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_s = build_mlp(dims=[state_dim, *dims], if_raw_out=False)  # network of encoded state
        self.net_a = build_mlp(dims=[dims[-1], action_dim * 2])  # the average and log_std of action

        init_with_orthogonal(self.net_a[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        s_enc = self.net_s(state)  # encoded state
        a_avg = self.net_a(s_enc)[:, :self.action_dim]
        return a_avg.tanh()  # action

    def get_action(self, state):
        state = self.state_norm(state)
        s_enc = self.net_s(state)  # encoded state
        a_avg, a_std_log = self.net_a(s_enc).chunk(2, dim=1)
        a_std = a_std_log.clamp(-16, 2).exp()

        dist = torch.distributions.normal.Normal(a_avg, a_std)
        return dist.rsample().tanh()  # action (re-parameterize)

    def get_action_logprob(self, state):
        state = self.state_norm(state)
        s_enc = self.net_s(state)  # encoded state
        a_avg, a_std_log = self.net_a(s_enc).chunk(2, dim=1)
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
        s_enc = self.net_s(state)  # encoded state
        a_avg, a_std_log = self.net_a(s_enc).chunk(2, dim=1)
        a_std = a_std_log.clamp(-16, 2).exp()

        dist = torch.distributions.normal.Normal(a_avg, a_std)
        action = dist.rsample()

        logprob = dist.log_prob(a_avg)
        logprob -= 2 * (math.log(2) - action - self.soft_plus(action * -2))  # fix logprob using SoftPlus
        return action.tanh(), logprob.sum(1)
