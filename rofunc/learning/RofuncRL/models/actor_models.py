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
import transformers
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


class ActorDTrans(nn.Module):
    def __init__(self, cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 state_encoder: Optional[nn.Module] = EmptyEncoder()):
        super().__init__()

        self.cfg = cfg
        self.action_dim = get_space_dim(action_space)
        self.n_embd = cfg.actor.n_embd
        self.max_ep_len = cfg.actor.max_episode_steps

        # state encoder
        self.state_encoder = state_encoder
        if isinstance(self.state_encoder, EmptyEncoder):
            self.state_dim = get_space_dim(observation_space)
        else:
            self.state_dim = self.state_encoder.output_dim

        gpt_config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=self.n_embd,
            n_layer=self.cfg.actor.n_layer,
            n_head=self.cfg.actor.n_head,
            n_inner=self.n_embd * 4,
            activation_function=self.cfg.actor.activation_function,
            resid_pdrop=self.cfg.actor.dropout,
            attn_pdrop=self.cfg.actor.dropout,
            n_positions=1024
        )

        self.embed_timestep = nn.Embedding(self.max_ep_len, self.n_embd)
        self.embed_return = torch.nn.Linear(1, self.n_embd)
        self.embed_state = torch.nn.Linear(self.state_dim, self.n_embd)
        self.embed_action = torch.nn.Linear(self.action_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)

        self.backbone_net = transformers.GPT2Model(gpt_config)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(self.n_embd, self.state_dim)
        self.predict_action = nn.Sequential(*([nn.Linear(self.n_embd, self.action_dim)] +
                                              ([nn.Tanh()] if self.cfg.use_action_out_tanh else [])))
        self.predict_return = torch.nn.Linear(self.n_embd, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # state encoder
        states = self.state_encoder(states)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1
                                     ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.n_embd)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack((attention_mask, attention_mask, attention_mask), dim=1
                                             ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.backbone_net(inputs_embeds=stacked_inputs,
                                                attention_mask=stacked_attention_mask)
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.n_embd).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:, 2])  # predict next return given state and action
        state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state

        return state_preds, action_preds, return_preds


if __name__ == '__main__':
    from omegaconf import DictConfig

    cfg = DictConfig({'use_init': True, 'action_scale': 1.0, 'use_log_std_clip': True, 'log_std_clip_min': -20,
                      'log_std_clip_max': 2, 'use_action_clip': True, 'action_clip': 1.0, 'actor':
                          {'type': "Gaussian", 'mlp_hidden_dims': [512, 256, 128], 'mlp_activation': "elu",
                           'use_lstm': False, 'lstm_cell_size': 256, 'lstm_use_prev_action': False, 'max_seq_len': 20}})
    model = ActorPPO_Gaussian(cfg=cfg, observation_space=3, action_space=1)
    print(model)
