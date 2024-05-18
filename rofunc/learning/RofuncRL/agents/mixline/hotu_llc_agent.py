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

from copy import deepcopy
from typing import Callable, Union, Tuple, Optional

import gym
import gymnasium
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

import rofunc as rf
from rofunc.learning.RofuncRL.agents.base_agent import BaseAgent
from rofunc.learning.RofuncRL.agents.mixline.ase_agent import ASEAgent
from rofunc.learning.RofuncRL.models.base_models import BaseMLP
from rofunc.learning.RofuncRL.models.critic_models import Critic
from rofunc.learning.RofuncRL.processors.standard_scaler import RunningStandardScaler
from rofunc.learning.RofuncRL.utils.memory import Memory


class HOTULLCAgent(ASEAgent):
    """
    HOTU - low-level controller agent for learning human and robots motion priors from unstructured motion data
    """
    def __init__(self,
                 cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 experiment_dir: Optional[str] = None,
                 rofunc_logger: Optional[rf.logger.BeautyLogger] = None,
                 amp_observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 motion_dataset: Optional[Union[Memory, Tuple[Memory]]] = None,
                 replay_buffer: Optional[Union[Memory, Tuple[Memory]]] = None,
                 collect_reference_motions: Optional[Callable[[int], torch.Tensor]] = None):
        """
        :param cfg: Configuration
        :param observation_space: Observation space
        :param action_space: Action space
        :param memory: Memory for storing transitions
        :param device: Device on which the torch tensor is allocated
        :param experiment_dir: Directory where experiment outputs are saved
        :param rofunc_logger: Rofunc logger
        :param amp_observation_space: cfg["env"]["numAMPObsSteps"] * NUM_AMP_OBS_PER_STEP
        :param motion_dataset: Motion dataset
        :param replay_buffer: Replay buffer
        :param collect_reference_motions: Function for collecting reference motions
        """
        if not isinstance(amp_observation_space, list):
            amp_observation_space = [amp_observation_space]

        self.num_parts = len(amp_observation_space)
        self.whole_amp_obs_space = amp_observation_space  # Whole AMP observation space

        self.motion_dataset_list = []
        self.replay_buffer_list = []
        for i in range(self.num_parts):
            self.motion_dataset_list.append(deepcopy(motion_dataset))
            self.replay_buffer_list.append(deepcopy(replay_buffer))

        super().__init__(cfg, observation_space, action_space, memory, device,
                         experiment_dir, rofunc_logger, amp_observation_space[0], motion_dataset, replay_buffer,
                         collect_reference_motions)  # Initialize the first part

        self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "log_prob", "values",
                               "returns", "advantages", "next_values", "ase_latents"]
        for i in range(self.num_parts):
            self._tensors_names.append(f"amp_states_{i}")
            self.memory.create_tensor(name=f"amp_states_{i}", size=amp_observation_space[i], dtype=torch.float32)
            # self.memory.create_tensor(name=f"ase_latents_{i}", size=self._ase_latent_dim, dtype=torch.float32)

        self._build_decompose_model()

    def _set_up(self):
        for i in range(self.num_parts):
            self.motion_dataset_list[i].create_tensor(name="states", size=self.whole_amp_obs_space[i],
                                                      dtype=torch.float32)
            self.replay_buffer_list[i].create_tensor(name="states", size=self.whole_amp_obs_space[i],
                                                     dtype=torch.float32)
        super()._set_up()

    def _initialize_motion_dataset(self):
        if self.collect_reference_motions is not None:
            for i in range(self.num_parts):
                for _ in range(math.ceil(self.motion_dataset.memory_size / self._amp_batch_size)):
                    self.motion_dataset_list[i].add_samples(
                        states=self.collect_reference_motions(self._amp_batch_size)[i])

    def _build_decompose_model(self):
        self.discriminator_list = [self.discriminator]
        self.encoder_list = [self.encoder]
        self._amp_state_preprocessor_kwargs_list = [self._amp_state_preprocessor_kwargs]
        self.optimizer_disc_list = [self.optimizer_disc]
        self.optimizer_enc_list = [self.optimizer_enc]
        self.scheduler_disc_list = [self.scheduler_disc]
        self.scheduler_enc_list = [self.scheduler_enc]
        self._amp_state_preprocessor_list = [self._amp_state_preprocessor]

        for i in range(1, self.num_parts):
            disc_model = Critic(cfg=self.cfg.Model,
                                observation_space=self.whole_amp_obs_space[i],
                                action_space=self.action_space,
                                state_encoder=self.se,
                                cfg_name='discriminator').to(self.device)
            enc_model = BaseMLP(cfg=self.cfg.Model,
                                input_dim=self.whole_amp_obs_space[i].shape[0],
                                output_dim=self._ase_latent_dim,
                                cfg_name='encoder').to(self.device)
            self.rofunc_logger.module(f"Discriminator model {i}: {disc_model}")
            self.rofunc_logger.module(f"Encoder model {i}: {enc_model}")

            amp_state_pre_kwargs = {"size": self.whole_amp_obs_space[i], "device": self.device}
            optimizer_disc = torch.optim.Adam(disc_model.parameters(), lr=self._lr_d, eps=self._adam_eps)
            optimizer_enc = torch.optim.Adam(enc_model.parameters(), lr=self._lr_e, eps=self._adam_eps)
            amp_state_preprocessor = RunningStandardScaler(**amp_state_pre_kwargs)

            if self._lr_scheduler is not None:
                scheduler_disc = self._lr_scheduler(optimizer_disc, **self._lr_scheduler_kwargs)
                scheduler_enc = self._lr_scheduler(optimizer_enc, **self._lr_scheduler_kwargs)
                self.scheduler_disc_list.append(scheduler_disc)
                self.scheduler_enc_list.append(scheduler_enc)

            self.discriminator_list.append(disc_model)
            self.encoder_list.append(enc_model)
            self._amp_state_preprocessor_kwargs_list.append(amp_state_pre_kwargs)
            self.optimizer_disc_list.append(optimizer_disc)
            self.optimizer_enc_list.append(optimizer_enc)
            self._amp_state_preprocessor_list.append(amp_state_preprocessor)

    def store_transition(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor,
                         rewards: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor,
                         infos: torch.Tensor):
        if self._current_states is not None:
            states = self._current_states

        BaseAgent.store_transition(self, states=states, actions=actions, next_states=next_states,
                                   rewards=rewards, terminated=terminated, truncated=truncated, infos=infos)

        # reward shaping
        if self._rewards_shaper is not None:
            rewards = self._rewards_shaper(rewards)

        # compute values
        values = self.value(self._state_preprocessor(torch.hstack((states, self._ase_latents))))
        # values = self.value(self._state_preprocessor(states))
        values = self._value_preprocessor(values, inverse=True)

        next_values = self.value(self._state_preprocessor(torch.hstack((next_states, self._ase_latents))))
        # next_values = self.value(self._state_preprocessor(next_states))
        next_values = self._value_preprocessor(next_values, inverse=True)
        next_values *= infos['terminate'].view(-1, 1).logical_not()

        # storage transition in memory
        amp_states_dict = {f"amp_states_{i}": infos[f"amp_obs_{i}"] for i in range(self.num_parts)}
        self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                terminated=terminated, truncated=truncated, log_prob=self._current_log_prob,
                                values=values, next_values=next_values, ase_latents=self._ase_latents,
                                **amp_states_dict)

    def update_net(self):
        """
        Update the network
        """
        '''Compute combined rewards'''
        rewards = self.memory.get_tensor_by_name("rewards")
        ase_latents = self.memory.get_tensor_by_name("ase_latents")

        amp_states_list = []
        for i in range(self.num_parts):
            # update dataset of reference motions
            self.motion_dataset_list[i].add_samples(states=self.collect_reference_motions(self._amp_batch_size)[i])
            amp_states_list.append(self.memory.get_tensor_by_name(f"amp_states_{i}"))

        with torch.no_grad():
            # Compute style reward from discriminator
            amp_logits_list = []
            for i in range(self.num_parts):
                amp_logits_list.append(
                    self.discriminator_list[i](self._amp_state_preprocessor_list[i](amp_states_list[i])))

            style_rewards = 1
            for i in range(self.num_parts):
                if self._least_square_discriminator:
                    style_rewards_tmp = torch.maximum(
                        torch.tensor(1 - 0.25 * torch.square(1 - amp_logits_list[i])),
                        torch.tensor(0.0001, device=self.device))
                else:
                    style_rewards_tmp = -torch.log(
                        torch.maximum(torch.tensor(1 - 1 / (1 + torch.exp(-amp_logits_list[i]))),
                                      torch.tensor(0.0001, device=self.device)))
                style_rewards *= style_rewards_tmp
            style_rewards *= self._discriminator_reward_scale

            # Compute encoder reward
            enc_reward = 1
            for i in range(self.num_parts):
                if self.encoder_list[i] is self.discriminator_list[i]:
                    enc_output = self.encoder_list[i].get_enc(
                        self._amp_state_preprocessor_list[i](amp_states_list[i]))
                else:
                    enc_output = self.encoder_list[i](
                        self._amp_state_preprocessor_list[i](amp_states_list[i]))
                enc_output = torch.nn.functional.normalize(enc_output, dim=-1)
                enc_reward_tmp = torch.clamp_min(torch.sum(enc_output * ase_latents, dim=-1, keepdim=True), 0.0)
                # enc_reward_tmp *= self._enc_reward_scale
                enc_reward *= enc_reward_tmp
            enc_reward *= self._enc_reward_scale

        combined_rewards = (self._task_reward_weight * rewards
                            + self._style_reward_weight * style_rewards
                            + self._enc_reward_weight * enc_reward)

        '''Compute Generalized Advantage Estimator (GAE)'''
        values = self.memory.get_tensor_by_name("values")
        next_values = self.memory.get_tensor_by_name("next_values")

        advantage = 0
        advantages = torch.zeros_like(combined_rewards)
        not_dones = self.memory.get_tensor_by_name("terminated").logical_not()
        memory_size = combined_rewards.shape[0]

        # advantages computation
        for i in reversed(range(memory_size)):
            advantage = combined_rewards[i] - values[i] + self._discount * (
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

        sampled_motion_batches_list = []
        sampled_replay_batches_list = []
        for i in range(self.num_parts):
            sampled_motion_batches = self.motion_dataset_list[i].sample(names=["states"],
                                                                        batch_size=self.memory.memory_size * self.memory.num_envs,
                                                                        mini_batches=self._mini_batch_size)

            if len(self.replay_buffer):
                sampled_replay_batches = self.replay_buffer_list[i].sample(names=["states"],
                                                                           batch_size=self.memory.memory_size * self.memory.num_envs,
                                                                           mini_batches=self._mini_batch_size)
            else:
                sampled_replay_batches = [[batches[self._tensors_names.index(f"amp_states_{i}")]] for batches in
                                          sampled_batches]
            sampled_motion_batches_list.append(sampled_motion_batches)
            sampled_replay_batches_list.append(sampled_replay_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        # cumulative_discriminator_loss = 0
        # cumulative_encoder_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            # mini-batches loop
            for i, sampled_tensors in enumerate(sampled_batches):
                (sampled_states, sampled_actions, sampled_rewards, samples_next_states, samples_terminated,
                 sampled_log_prob, sampled_values, sampled_returns, sampled_advantages, _,
                 sampled_ase_latents) = sampled_tensors[:11]

                sampled_amp_states_list = sampled_tensors[11:]
                assert len(sampled_amp_states_list) == self.num_parts

                sampled_states = self._state_preprocessor(torch.hstack((sampled_states, sampled_ase_latents)),
                                                          train=True)
                # sampled_states = self._state_preprocessor(sampled_states, train=True)
                res_dict = self.policy(sampled_states, sampled_actions)
                log_prob_now = res_dict["log_prob"]

                # compute entropy loss
                entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy().mean()

                # compute policy loss
                ratio = torch.exp(log_prob_now - sampled_log_prob)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip,
                                                                    1.0 + self._ratio_clip)

                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # compute value loss
                predicted_values = self.value(sampled_states)

                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(predicted_values - sampled_values,
                                                                   min=-self._value_clip,
                                                                   max=self._value_clip)
                value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                discriminator_loss_list = []
                enc_loss_list = []
                for part_i in range(self.num_parts):
                    discriminator_loss, enc_loss = self._calc_disc_enc_loss(sampled_amp_states_list[part_i],
                                                                            sampled_replay_batches_list[part_i],
                                                                            sampled_motion_batches_list[part_i],
                                                                            sampled_ase_latents,
                                                                            self.discriminator_list[part_i],
                                                                            self.encoder_list[part_i],
                                                                            self._amp_state_preprocessor_list[part_i],
                                                                            i)
                    discriminator_loss_list.append(discriminator_loss)
                    enc_loss_list.append(enc_loss)

                # if self._enable_amp_diversity_bonus():
                #     diversity_loss = self._diversity_loss(batch_dict['obs'], mu, batch_dict['ase_latents'])
                #     diversity_loss = torch.sum(rand_action_mask * diversity_loss) / rand_action_sum
                #     loss += self._amp_diversity_bonus * diversity_loss
                #     a_info['amp_diversity_loss'] = diversity_loss

                '''Update networks'''
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

                for part_i in range(self.num_parts):
                    self._update_disc_enc(discriminator_loss_list[part_i], enc_loss_list[part_i],
                                          self.discriminator_list[part_i], self.encoder_list[part_i],
                                          self.optimizer_disc_list[part_i], self.optimizer_enc_list[part_i])

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                # cumulative_discriminator_loss += discriminator_loss.item()
                # cumulative_encoder_loss += enc_loss.item()

            # update learning rate
            if self._lr_scheduler:
                self.scheduler_policy.step()
                self.scheduler_value.step()

                for part_i in range(self.num_parts):
                    self.scheduler_disc_list[part_i].step()
                    if self.encoder_list[part_i] is not self.discriminator_list[part_i]:
                        self.scheduler_enc_list[part_i].step()

        # update AMP replay buffer
        for i in range(self.num_parts):
            self.replay_buffer_list[i].add_samples(states=amp_states_list[i].view(-1, amp_states_list[i].shape[-1]))

        # record data
        self.track_data("Info / Combined rewards", combined_rewards.mean().cpu())
        self.track_data("Info / Style rewards", style_rewards.mean().cpu())
        self.track_data("Info / Encoder rewards", enc_reward.mean().cpu())
        self.track_data("Info / Task rewards", rewards.mean().cpu())

        self.track_data("Loss / Policy loss",
                        cumulative_policy_loss / (self._learning_epochs * self._mini_batch_size))
        self.track_data("Loss / Value loss",
                        cumulative_value_loss / (self._learning_epochs * self._mini_batch_size))
        # self.track_data("Loss / Discriminator loss",
        #                 cumulative_discriminator_loss / (self._learning_epochs * self._mini_batch_size))
        # self.track_data("Loss / Encoder loss",
        #                 cumulative_encoder_loss / (self._learning_epochs * self._mini_batch_size))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss",
                            cumulative_entropy_loss / (self._learning_epochs * self._mini_batch_size))
        if self._lr_scheduler:
            self.track_data("Learning / Learning rate (policy)", self.scheduler_policy.get_last_lr()[0])
        self.track_data("Learning / Learning rate (value)", self.scheduler_value.get_last_lr()[0])
        self.track_data("Learning / Learning rate (discriminator)", self.scheduler_disc.get_last_lr()[0])
        if self.encoder is not self.discriminator:
            self.track_data("Learning / Learning rate (encoder)", self.scheduler_enc.get_last_lr()[0])

    def _calc_disc_enc_loss(self, sampled_amp_states, sampled_replay_batches, sampled_motion_batches,
                            sampled_ase_latents, discriminator, encoder, amp_state_preprocessor, i):
        # compute discriminator loss
        if self._discriminator_batch_size:
            sampled_amp_states_batch = amp_state_preprocessor(
                sampled_amp_states[0:self._discriminator_batch_size], train=True)
            sampled_amp_replay_states = amp_state_preprocessor(
                sampled_replay_batches[i][0][0:self._discriminator_batch_size], train=True)
            sampled_amp_motion_states = amp_state_preprocessor(
                sampled_motion_batches[i][0][0:self._discriminator_batch_size], train=True)
        else:
            sampled_amp_states_batch = amp_state_preprocessor(sampled_amp_states, train=True)
            sampled_amp_replay_states = amp_state_preprocessor(sampled_replay_batches[i][0],
                                                               train=True)
            sampled_amp_motion_states = amp_state_preprocessor(sampled_motion_batches[i][0],
                                                               train=True)

        sampled_amp_motion_states.requires_grad_(True)
        amp_logits = discriminator(sampled_amp_states_batch)
        amp_replay_logits = discriminator(sampled_amp_replay_states)
        amp_motion_logits = discriminator(sampled_amp_motion_states)
        amp_cat_logits = torch.cat([amp_logits, amp_replay_logits], dim=0)

        # discriminator prediction loss
        if self._least_square_discriminator:
            discriminator_loss = 0.5 * (
                    F.mse_loss(amp_cat_logits, -torch.ones_like(amp_cat_logits), reduction='mean')
                    + F.mse_loss(amp_motion_logits, torch.ones_like(amp_motion_logits),
                                 reduction='mean'))
        else:
            discriminator_loss = 0.5 * (
                    nn.BCEWithLogitsLoss()(amp_cat_logits, torch.zeros_like(amp_cat_logits))
                    + nn.BCEWithLogitsLoss()(amp_motion_logits,
                                             torch.ones_like(amp_motion_logits)))

        # discriminator logit regularization
        if self._discriminator_logit_regularization_scale:
            logit_weights = torch.flatten(list(discriminator.modules())[-1].weight)
            discriminator_loss += self._discriminator_logit_regularization_scale * torch.sum(
                torch.square(logit_weights))

        # discriminator gradient penalty TODO: check whether this is used
        if self._discriminator_gradient_penalty_scale:
            amp_motion_gradient = torch.autograd.grad(amp_motion_logits,
                                                      sampled_amp_motion_states,
                                                      grad_outputs=torch.ones_like(
                                                          amp_motion_logits),
                                                      create_graph=True,
                                                      retain_graph=True,
                                                      only_inputs=True)
            gradient_penalty = torch.sum(torch.square(amp_motion_gradient[0]), dim=-1).mean()
            discriminator_loss += self._discriminator_gradient_penalty_scale * gradient_penalty

        # discriminator weight decay
        if self._discriminator_weight_decay_scale:
            weights = [torch.flatten(module.weight) for module in discriminator.modules()
                       if isinstance(module, torch.nn.Linear)]
            weight_decay = torch.sum(torch.square(torch.cat(weights, dim=-1)))
            discriminator_loss += self._discriminator_weight_decay_scale * weight_decay

        discriminator_loss *= self._discriminator_loss_scale

        # encoder loss
        if encoder is discriminator:
            enc_output = encoder.get_enc(amp_state_preprocessor(sampled_amp_states))
        else:
            enc_output = encoder(amp_state_preprocessor(sampled_amp_states))
        enc_output = torch.nn.functional.normalize(enc_output, dim=-1)
        enc_err = -torch.sum(enc_output * sampled_ase_latents, dim=-1, keepdim=True)
        enc_loss = torch.mean(enc_err)

        # encoder gradient penalty
        if self._enc_gradient_penalty_scale:
            enc_obs_grad = torch.autograd.grad(enc_err,
                                               sampled_ase_latents,
                                               grad_outputs=torch.ones_like(enc_err),
                                               create_graph=True,
                                               retain_graph=True,
                                               only_inputs=True)
            gradient_penalty = torch.sum(torch.square(enc_obs_grad[0]), dim=-1).mean()
            enc_loss += self._enc_gradient_penalty_scale * gradient_penalty

        # encoder weight decay
        if self._enc_weight_decay_scale:
            weights = [torch.flatten(module.weight) for module in encoder.modules()
                       if isinstance(module, torch.nn.Linear)]
            weight_decay = torch.sum(torch.square(torch.cat(weights, dim=-1)))
            enc_loss += self._enc_weight_decay_scale * weight_decay

        return discriminator_loss, enc_loss

    def _update_disc_enc(self, discriminator_loss, enc_loss, discriminator, encoder, optimizer_disc, optimizer_enc):
        # Update discriminator network
        optimizer_disc.zero_grad()
        if encoder is discriminator:
            (discriminator_loss + enc_loss).backward()
        else:
            discriminator_loss.backward()
        if self._grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(), self._grad_norm_clip)
        optimizer_disc.step()

        # Update encoder network
        if encoder is not discriminator:
            optimizer_enc.zero_grad()
            enc_loss.backward()
            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(encoder.parameters(), self._grad_norm_clip)
            optimizer_enc.step()
