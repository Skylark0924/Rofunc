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

from typing import Union, Tuple, Optional

import gym
import gymnasium
import torch
import torch.nn as nn
import transformers
from omegaconf import DictConfig
from torch import Tensor

from rofunc.config.utils import omegaconf_to_dict
from rofunc.learning.RofuncRL.models.base_models import BaseMLP
from rofunc.learning.RofuncRL.models.utils import init_layers, get_space_dim
from rofunc.learning.RofuncRL.state_encoders.base_encoders import EmptyEncoder


class ASEDiscEnc(BaseMLP):
    def __init__(self, cfg: DictConfig,
                 input_dim: int,
                 enc_output_dim: int,
                 disc_output_dim: int,
                 cfg_name: str):
        super().__init__(cfg, input_dim, disc_output_dim, cfg_name)
        self.encoder_layer = nn.Linear(self.mlp_hidden_dims[-1], enc_output_dim)
        self.disc_layer = self.output_net
        if self.cfg.use_init:
            init_layers(self.encoder_layer, gain=1.0)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)  # Same as self.get_disc(x)

    def get_enc(self, x: Tensor) -> Tensor:
        x = self.backbone_net(x)
        x = self.encoder_layer(x)
        return x

    def get_disc(self, x: Tensor) -> Tensor:
        x = self.backbone_net(x)
        x = self.disc_layer(x)
        return x


class DTrans(nn.Module):
    def __init__(self, cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 state_encoder: Optional[nn.Module] = EmptyEncoder(),
                 cfg_name='actor'):
        super().__init__()

        self.cfg = cfg
        self.cfg_dict = omegaconf_to_dict(self.cfg)
        self.action_dim = get_space_dim(action_space)
        self.n_embd = self.cfg_dict[cfg_name]["n_embd"]
        self.max_ep_len = self.cfg_dict[cfg_name]["max_episode_steps"]

        # state encoder
        self.state_encoder = state_encoder
        if isinstance(self.state_encoder, EmptyEncoder):
            self.state_dim = get_space_dim(observation_space)
        else:
            self.state_dim = self.state_encoder.output_dim

        gpt_config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=self.n_embd,
            n_layer=self.cfg_dict[cfg_name]["n_layer"],
            n_head=self.cfg_dict[cfg_name]["n_head"],
            n_inner=self.n_embd * 4,
            activation_function=self.cfg_dict[cfg_name]["activation_function"],
            resid_pdrop=self.cfg_dict[cfg_name]["dropout"],
            attn_pdrop=self.cfg_dict[cfg_name]["dropout"],
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
        """
        :param states: [B, T, C, H, W] or [B, T, Value]
        :param actions: [B, T, Action]
        :param rewards: [B, T, 1]
        :param returns_to_go: [B, T, 1]
        :param timesteps: [B, T]
        :param attention_mask: [B, T]
        :return:
        """
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # state encoder
        states = self.state_encoder(states)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)  # [B, T, n_embd]
        action_embeddings = self.embed_action(actions)  # [B, T, n_embd]
        returns_embeddings = self.embed_return(returns_to_go)  # [B, T, n_embd]
        time_embeddings = self.embed_timestep(timesteps)  # [B, T, n_embd]

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        # [B, 3, T, n_embd] -> [B, T, 3, n_embd] -> [B, T*3, n_embd]
        stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1
                                     ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.n_embd)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack((attention_mask, attention_mask, attention_mask), dim=1
                                             ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.backbone_net(inputs_embeds=stacked_inputs, attention_mask=stacked_attention_mask)
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.n_embd).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:, 2])  # predict next return given state and action
        state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state

        return state_preds, action_preds, return_preds
