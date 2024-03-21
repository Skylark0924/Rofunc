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

from typing import Union, Tuple, Optional

import gym
import gymnasium
import rofunc as rf
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from rofunc.learning.RofuncRL.agents.base_agent import BaseAgent
from rofunc.learning.RofuncRL.agents.online.ppo_agent import PPOAgent
from rofunc.learning.RofuncRL.processors.schedulers import KLAdaptiveRL
from rofunc.learning.RofuncRL.utils.memory import Memory


class PhysHOIAgent(PPOAgent, BaseAgent):
    """
    PhysHOI agent
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
        :param cfg: Configuration
        :param observation_space: Observation space
        :param action_space: Action space
        :param memory: Memory for storing transitions
        :param device: Device on which the torch tensor is allocated
        :param experiment_dir: Directory where experiment outputs are saved
        :param rofunc_logger: Rofunc logger
        """
        super().__init__(
            cfg,
            observation_space,
            action_space,
            memory,
            device,
            experiment_dir,
            rofunc_logger,
        )
        if hasattr(cfg.Model, "state_encoder"):
            img_channel = int(self.cfg.Model.state_encoder.inp_channels)
            img_size = int(self.cfg.Model.state_encoder.image_size)
            state_tensor_size = (img_channel, img_size, img_size)
            kd = True
        else:
            state_tensor_size = self.observation_space
            kd = False
        self.memory.create_tensor(name="next_states", size=state_tensor_size, dtype=torch.float32, keep_dimensions=kd)
        self.memory.create_tensor(name="next_values", size=1, dtype=torch.float32)

        self._rewards_shaper = self.cfg.get("Agent", {}).get("rewards_shaper", lambda rewards: rewards * 1)

        self.scaler = torch.cuda.amp.GradScaler(enabled=False)

    def store_transition(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor,
                         rewards: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, infos: torch.Tensor):
        BaseAgent.store_transition(self, states=states, actions=actions, next_states=next_states, rewards=rewards,
                                   terminated=terminated, truncated=truncated, infos=infos)
        self._current_next_states = next_states

        # reward shaping
        if self._rewards_shaper is not None:
            rewards = self._rewards_shaper(rewards)

        # compute values
        if self.cfg.Model.use_same_model:
            values = self.value.get_value(self._state_preprocessor(states))
        else:
            values = self.value(self._state_preprocessor(states))
        values = self._value_preprocessor(values, inverse=True)

        if self.cfg.Model.use_same_model:
            next_values = self.value.get_value(self._state_preprocessor(next_states))
        else:
            next_values = self.value(self._state_preprocessor(next_states))
        next_values = self._value_preprocessor(next_values, inverse=True)
        next_values *= infos['terminate'].view(-1, 1).logical_not()

        # storage transition in memory
        self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                terminated=terminated, truncated=truncated, log_prob=self._current_log_prob,
                                values=values, next_values=next_values)

    def bound_loss(self, mu):
        soft_bound = 1.0
        mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
        mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
        b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)

        return b_loss

    def update_net(self):
        """
        Update the network
        """
        '''Compute Generalized Advantage Estimator (GAE)'''
        values = self.memory.get_tensor_by_name("values")
        next_values = self.memory.get_tensor_by_name("next_values")

        advantage = 0
        advantages = torch.zeros_like(self.memory.get_tensor_by_name("rewards"))
        not_dones = self.memory.get_tensor_by_name("terminated").logical_not()
        memory_size = self.memory.get_tensor_by_name("rewards").shape[0]

        # advantages computation
        for i in reversed(range(memory_size)):
            advantage = self.memory.get_tensor_by_name("rewards")[i] - values[i] + self._discount * (
                    next_values[i] + self._td_lambda * not_dones[i] * advantage)
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
            for i, (
                    sampled_states, sampled_actions, sampled_dones, sampled_log_prob, sampled_values,
                    sampled_returns,
                    sampled_advantages) in enumerate(sampled_batches):
                sampled_states = self._state_preprocessor(sampled_states, train=not epoch)
                _, log_prob_now = self.policy(sampled_states, sampled_actions)

                # compute approximate KL divergence
                with torch.no_grad():
                    ratio = log_prob_now - sampled_log_prob
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)

                # early stopping with KL divergence
                if self._kl_threshold and kl_divergence > self._kl_threshold:
                    break

                # compute entropy loss
                entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy().mean()

                # compute policy loss
                ratio = torch.exp(log_prob_now - sampled_log_prob)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip,
                                                                    1.0 + self._ratio_clip)

                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # compute value loss
                if self.cfg.Model.use_same_model:
                    predicted_values = self.value.get_value(sampled_states)
                else:
                    predicted_values = self.value(sampled_states)

                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(predicted_values - sampled_values,
                                                                   min=-self._value_clip,
                                                                   max=self._value_clip)
                value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # compute bound loss
                b_loss = 10 * self.bound_loss(sampled_actions)
                b_loss = torch.mean(b_loss)

                loss = policy_loss + value_loss + b_loss

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # if self.policy is self.value:
                #     # optimization step
                #     self.optimizer.zero_grad()
                #     (policy_loss + entropy_loss + value_loss + b_loss).backward()
                #     if self._grad_norm_clip > 0:
                #         nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                #     self.optimizer.step()
                # else:
                #     # Update policy network
                #     self.optimizer_policy.zero_grad()
                #     (policy_loss + entropy_loss + b_loss).backward()
                #     if self._grad_norm_clip > 0:
                #         nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                #     self.optimizer_policy.step()
                #
                #     # Update value network
                #     self.optimizer_value.zero_grad()
                #     value_loss.backward()
                #     if self._grad_norm_clip > 0:
                #         nn.utils.clip_grad_norm_(self.value.parameters(), self._grad_norm_clip)
                #     self.optimizer_value.step()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # # update learning rate
            # if self._lr_scheduler:
            #     if self.policy is self.value:
            #         if isinstance(self.scheduler, KLAdaptiveRL):
            #             self.scheduler.step(torch.tensor(kl_divergences).mean())
            #         else:
            #             self.scheduler.step()
            #     else:
            #         if isinstance(self.scheduler_policy, KLAdaptiveRL):
            #             self.scheduler_policy.step(torch.tensor(kl_divergences).mean())
            #         else:
            #             self.scheduler_policy.step()
            #         if isinstance(self.scheduler_value, KLAdaptiveRL):
            #             self.scheduler_value.step(torch.tensor(kl_divergences).mean())
            #         else:
            #             self.scheduler_value.step()

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
