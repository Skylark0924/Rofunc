import math
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .base_model import BaseActor, build_mlp, init_with_orthogonal, activation_func


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


class ActorPPO(BaseActor):
    def __init__(self, cfg: DictConfig, observation_space: int, action_space: int):
        state_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.mlp_hidden_dims = cfg.mlp_hidden_dims
        self.mlp_activation = activation_func(cfg.mlp_activation)

        self.net = build_mlp(dims=[state_dim, *self.mlp_hidden_dims, action_dim], hidden_activation=self.mlp_activation)

        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        return self.net(state).tanh()  # action.tanh()

    def act(self, state: Tensor, deterministic=False) -> (Tensor, Tensor):  # for exploration
        state = self.state_norm(state)
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)

        if deterministic:
            action = dist.mode
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(1)
        return action, log_prob

    def get_log_prob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        state = self.state_norm(state)
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        log_prob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return log_prob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.tanh()


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
