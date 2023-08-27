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

import gym
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import Union, Tuple, Optional

import rofunc as rf
from rofunc.learning.RofuncRL.agents.base_agent import BaseAgent
from rofunc.learning.RofuncRL.models.actor_models import ActorPPO_Gaussian
from rofunc.learning.RofuncRL.models.critic_models import Critic
from rofunc.learning.RofuncRL.processors.schedulers import KLAdaptiveRL
from rofunc.learning.RofuncRL.processors.standard_scaler import RunningStandardScaler
# from rofunc.learning.RofuncRL.utils.skrl_utils import Policy, Value
from rofunc.learning.RofuncRL.processors.standard_scaler import empty_preprocessor
from rofunc.learning.RofuncRL.utils.memory import Memory


class A2CAgent(BaseAgent):
    """
    Advantage Actor Critic (A2C) agent \n
    "Asynchronous Methods for Deep Reinforcement Learning". Mnih et al. 2016. https://arxiv.org/abs/1602.01783 \n
    Rofunc documentation: https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/A2C.html
    """

    def __init__(self,
                 cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 experiment_dir: Optional[str] = None,
                 rofunc_logger: Optional[rf.logger.BeautyLogger] = None):
        """
        :param cfg: Configurations
        :param observation_space: Observation space
        :param action_space: Action space
        :param memory: Memory for storing transitions
        :param device: Device on which the torch tensor is allocated
        :param experiment_dir: Directory for storing experiment data
        :param rofunc_logger: Rofunc logger
        """
        super().__init__(cfg, observation_space, action_space, memory, device, experiment_dir, rofunc_logger)

        '''Define models for A2C'''
        self.policy = ActorPPO_Gaussian(cfg.Model, observation_space, action_space, self.se).to(self.device)
        self.value = Critic(cfg.Model, observation_space, action_space, self.se).to(self.device)
        # self.policy = Policy(observation_space, action_space, device, clip_actions=True).to(self.device)
        # self.value = Value(observation_space, action_space, device).to(self.device)
        self.models = {"policy": self.policy, "value": self.value}
        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        self.rofunc_logger.module(f"Policy model: {self.policy}")
        self.rofunc_logger.module(f"Value model: {self.value}")

        '''Create tensors in memory'''
        if hasattr(cfg.Model, "state_encoder"):
            img_channel = int(self.cfg.Model.state_encoder.inp_channels)
            img_size = int(self.cfg.Model.state_encoder.image_size)
            state_tensor_size = (img_channel, img_size, img_size)
            kd = True
        else:
            state_tensor_size = self.observation_space
            kd = False
        self.memory.create_tensor(name="states", size=state_tensor_size, dtype=torch.float32, keep_dimensions=kd)
        self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
        self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
        self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)
        # tensors sampled during training
        self._tensors_names = ["states", "actions", "terminated", "log_prob", "values", "returns", "advantages"]

        '''Get hyper-parameters from config'''
        self._discount = self.cfg.Agent.discount
        self._td_lambda = self.cfg.Agent.td_lambda
        self._learning_epochs = self.cfg.Agent.learning_epochs
        self._mini_batch_size = self.cfg.Agent.mini_batch_size
        self._lr_a = self.cfg.Agent.lr_a
        self._lr_c = self.cfg.Agent.lr_c
        self._lr_scheduler = self.cfg.get("Agent", {}).get("lr_scheduler", KLAdaptiveRL)
        self._lr_scheduler_kwargs = self.cfg.get("Agent", {}).get("lr_scheduler_kwargs", {'kl_threshold': 0.008})
        self._adam_eps = self.cfg.Agent.adam_eps
        self._use_gae = self.cfg.Agent.use_gae
        self._entropy_loss_scale = self.cfg.Agent.entropy_loss_scale
        self._grad_norm_clip = self.cfg.Agent.grad_norm_clip
        self._ratio_clip = self.cfg.Agent.ratio_clip
        self._value_clip = self.cfg.Agent.value_clip
        self._clip_predicted_values = self.cfg.Agent.clip_predicted_values
        self._kl_threshold = self.cfg.Agent.kl_threshold
        self._rewards_shaper = self.cfg.get("Agent", {}).get("rewards_shaper", lambda rewards: rewards * 0.01)
        self._state_preprocessor = RunningStandardScaler
        self._state_preprocessor_kwargs = self.cfg.get("Agent", {}).get("state_preprocessor_kwargs",
                                                                        {"size": observation_space, "device": device})
        self._value_preprocessor = RunningStandardScaler
        self._value_preprocessor_kwargs = self.cfg.get("Agent", {}).get("value_preprocessor_kwargs",
                                                                        {"size": 1, "device": device})

        '''Misc variables'''
        self._current_log_prob = None
        self._current_next_states = None

        self._set_up()

    def _set_up(self):
        """
        Set up optimizer, learning rate scheduler and state/value preprocessors
        """
        # Set up optimizer and learning rate scheduler
        if self.policy is self.value:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._lr_a)
            if self._lr_scheduler is not None:
                self.scheduler = self._lr_scheduler(self.optimizer, **self._lr_scheduler_kwargs)
            self.checkpoint_modules["optimizer"] = self.optimizer
        else:
            self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=self._lr_a, eps=self._adam_eps)
            self.optimizer_value = torch.optim.Adam(self.value.parameters(), lr=self._lr_c, eps=self._adam_eps)
            if self._lr_scheduler is not None:
                self.scheduler_policy = self._lr_scheduler(self.optimizer_policy, **self._lr_scheduler_kwargs)
                self.scheduler_value = self._lr_scheduler(self.optimizer_value, **self._lr_scheduler_kwargs)
            self.checkpoint_modules["optimizer_policy"] = self.optimizer_policy
            self.checkpoint_modules["optimizer_value"] = self.optimizer_value

        # set up preprocessors
        super()._set_up()

    def act(self, states: torch.Tensor, deterministic: bool = False):
        if not deterministic:
            # sample stochastic actions
            actions, log_prob = self.policy(self._state_preprocessor(states))
            self._current_log_prob = log_prob
        else:
            # choose deterministic actions for evaluation
            actions = self.policy(self._state_preprocessor(states)).detach()
            log_prob = None
        return actions, log_prob

    def store_transition(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor,
                         rewards: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, infos: torch.Tensor):
        super().store_transition(states=states, actions=actions, next_states=next_states, rewards=rewards,
                                 terminated=terminated, truncated=truncated, infos=infos)

        self._current_next_states = next_states

        # reward shaping
        if self._rewards_shaper is not None:
            rewards = self._rewards_shaper(rewards)

        # compute values
        values = self.value(self._state_preprocessor(states))
        values = self._value_preprocessor(values, inverse=True)

        # storage transition in memory
        self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                terminated=terminated, truncated=truncated, log_prob=self._current_log_prob,
                                values=values)

    def update_net(self):
        """
        Update the network
        """
        '''Compute Generalized Advantage Estimator (GAE)'''
        values = self.memory.get_tensor_by_name("values")

        with torch.no_grad():
            self.value.train(False)
            next_values = self.value(self._state_preprocessor(self._current_next_states.float()))
            self.value.train(True)
        next_values = self._value_preprocessor(next_values, inverse=True)

        advantage = 0
        advantages = torch.zeros_like(self.memory.get_tensor_by_name("rewards"))
        not_dones = self.memory.get_tensor_by_name("terminated").logical_not()
        memory_size = self.memory.get_tensor_by_name("rewards").shape[0]

        # advantages computation
        for i in reversed(range(memory_size)):
            next_values = values[i + 1] if i < memory_size - 1 else next_values
            advantage = self.memory.get_tensor_by_name("rewards")[i] - values[i] + self._discount * not_dones[i] * (
                    next_values + self._td_lambda * advantage)
            advantages[i] = advantage
        # returns computation
        values_target = advantages + values
        # advantage normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(values_target, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        '''Sample mini-batches from memory and update the network'''
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batch_size)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for i, (sampled_states, sampled_actions, sampled_dones, sampled_log_prob, sampled_values, sampled_returns,
                    sampled_advantages) in enumerate(sampled_batches):
                sampled_states = self._state_preprocessor(sampled_states, train=not epoch)
                _, log_prob_now = self.policy(sampled_states, sampled_actions)

                # compute approximate KL divergence
                with torch.no_grad():
                    ratio = log_prob_now - sampled_log_prob
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)

                # compute entropy loss
                entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy().mean()

                # compute policy loss
                policy_loss = -(sampled_advantages * log_prob_now).mean()

                # compute value loss
                predicted_values = self.value(sampled_states)

                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(predicted_values - sampled_values,
                                                                   min=-self._value_clip,
                                                                   max=self._value_clip)
                value_loss = F.mse_loss(sampled_returns, predicted_values)

                if self.policy is self.value:
                    # optimization step
                    self.optimizer.zero_grad()
                    (policy_loss + entropy_loss + value_loss).backward()
                    if self._grad_norm_clip > 0:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    self.optimizer.step()
                else:
                    # Update policy network
                    self.optimizer_policy.zero_grad()
                    (policy_loss + entropy_loss).backward()
                    if self._grad_norm_clip > 0:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    self.optimizer_policy.step()

                    # Update value network
                    self.optimizer_value.zero_grad()
                    value_loss.backward()
                    if self._grad_norm_clip > 0:
                        nn.utils.clip_grad_norm_(self.value.parameters(), self._grad_norm_clip)
                    self.optimizer_value.step()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._lr_scheduler:
                if self.policy is self.value:
                    if isinstance(self.scheduler, KLAdaptiveRL):
                        self.scheduler.step(torch.tensor(kl_divergences).mean())
                    else:
                        self.scheduler.step()
                else:
                    if isinstance(self.scheduler_policy, KLAdaptiveRL):
                        self.scheduler_policy.step(torch.tensor(kl_divergences).mean())
                    else:
                        self.scheduler_policy.step()
                    if isinstance(self.scheduler_value, KLAdaptiveRL):
                        self.scheduler_value.step(torch.tensor(kl_divergences).mean())
                    else:
                        self.scheduler_value.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batch_size))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batch_size))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss",
                            cumulative_entropy_loss / (self._learning_epochs * self._mini_batch_size))
        if self._lr_scheduler:
            if self.policy is self.value:
                self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
            else:
                self.track_data("Learning / Learning rate (policy)", self.scheduler_policy.get_last_lr()[0])
                self.track_data("Learning / Learning rate (value)", self.scheduler_value.get_last_lr()[0])
