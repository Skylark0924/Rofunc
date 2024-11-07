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

import collections
from typing import Union, Tuple, Optional

import gym
import gymnasium
import torch
from omegaconf import DictConfig

import rofunc as rf
from rofunc.learning.RofuncRL.agents.base_agent import BaseAgent
from rofunc.learning.RofuncRL.models.misc_models import DTrans


class DTransAgent(BaseAgent):
    """
    Decision Transformer (DTrans) Agent \n
    “Decision Transformer: Reinforcement Learning via Sequence Modeling”. Chen et al. 2021. https://arxiv.org/abs/2106.01345 \n
    Rofunc documentation: https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/DTrans.html \n
    """

    def __init__(self,
                 cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 device: Optional[Union[str, torch.device]] = None,
                 experiment_dir: Optional[str] = None,
                 rofunc_logger: Optional[rf.logger.BeautyLogger] = None):
        """
        :param cfg: Configurations
        :param observation_space: Observation space
        :param action_space: Action space
        :param device: Device on which the torch tensor is allocated
        :param experiment_dir: Directory for storing experiment data
        :param rofunc_logger: Rofunc logger
        """

        super().__init__(cfg, observation_space, action_space, None, device, experiment_dir, rofunc_logger)

        self.dtrans = DTrans(cfg.Model, observation_space, action_space, self.se).to(self.device)
        self.models = {"dtrans": self.dtrans}

        # checkpoint models
        self.checkpoint_modules["dtrans"] = self.dtrans
        self.rofunc_logger.module(f"DTrans model: {self.dtrans}")

        self.track_losses = collections.deque(maxlen=100)
        self.tracking_data = collections.defaultdict(list)
        self.checkpoint_best_modules = {"timestep": 0, "loss": 2 ** 31, "saved": False, "modules": {}}

        '''Get hyper-parameters from config'''
        self._td_lambda = self.cfg.Agent.td_lambda
        self._lr = self.cfg.Agent.lr
        self._adam_eps = self.cfg.Agent.adam_eps
        self._weight_decay = self.cfg.Agent.weight_decay
        self._max_seq_length = self.cfg.Trainer.max_seq_length

        self._set_up()

    def _set_up(self):
        """
        Set up optimizer, learning rate scheduler and state/value preprocessors
        """
        self.optimizer = torch.optim.AdamW(self.dtrans.parameters(), lr=self._lr, eps=self._adam_eps,
                                           weight_decay=self._weight_decay)
        if self._lr_scheduler is not None:
            self.scheduler = self._lr_scheduler(self.optimizer, **self._lr_scheduler_kwargs)
        self.checkpoint_modules["optimizer_policy"] = self.optimizer

        self.loss_fn = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2)

        # set up preprocessors
        super()._set_up()

    def act(self, states, actions, rewards, returns_to_go, timesteps):
        # we don't care about the past rewards in this model
        states = states.reshape(1, -1, self.dtrans.state_dim)
        actions = actions.reshape(1, -1, self.dtrans.action_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self._max_seq_length is not None:
            states = states[:, -self._max_seq_length:]
            actions = actions[:, -self._max_seq_length:]
            returns_to_go = returns_to_go[:, -self._max_seq_length:]
            timesteps = timesteps[:, -self._max_seq_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [torch.zeros(self._max_seq_length - states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self._max_seq_length - states.shape[1], self.dtrans.state_dim),
                             device=states.device), states], dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self._max_seq_length - actions.shape[1], self.dtrans.action_dim),
                             device=actions.device), actions], dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self._max_seq_length - returns_to_go.shape[1], 1),
                             device=returns_to_go.device), returns_to_go], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat([torch.zeros((timesteps.shape[0], self._max_seq_length - timesteps.shape[1]),
                                               device=timesteps.device), timesteps], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.dtrans(states, actions, None, returns_to_go, timesteps, attention_mask)

        return action_preds[0, -1]

    def update_net(self, batch):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = batch
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.dtrans.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(None, action_preds, None,
                            None, action_target, None)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dtrans.parameters(), .25)
        self.optimizer.step()

        # with torch.no_grad():
        #     self.diagnostics['training/action_error'] = torch.mean(
        #         (action_preds - action_target) ** 2).detach().cpu().item()

        # update learning rate
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

        # record data
        self.track_data("Loss", loss.item())
        self.track_data("Action_error", torch.mean((action_preds - action_target) ** 2).item())
