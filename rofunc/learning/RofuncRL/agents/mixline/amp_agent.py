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

import math
from typing import Callable, Union, Tuple, Optional, List

import gym
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

import rofunc as rf
from rofunc.learning.RofuncRL.agents.base_agent import BaseAgent
from rofunc.learning.RofuncRL.models.actor_models import ActorAMP
from rofunc.learning.RofuncRL.models.critic_models import Critic
from rofunc.learning.RofuncRL.processors.schedulers import KLAdaptiveRL
from rofunc.learning.RofuncRL.processors.standard_scaler import RunningStandardScaler
from rofunc.learning.RofuncRL.processors.standard_scaler import empty_preprocessor
from rofunc.learning.RofuncRL.utils.memory import Memory


class AMPAgent(BaseAgent):
    """
    Adversarial Motion Priors (AMP) agent \n
    “AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control”. Peng et al. 2021. https://arxiv.org/abs/2104.02180 \n
    Rofunc documentation: https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/AMP.html
    """

    def __init__(self,
                 cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space, List]],
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
        super().__init__(cfg, observation_space, action_space, memory, device, experiment_dir, rofunc_logger)

        self.amp_observation_space = amp_observation_space
        self.motion_dataset = motion_dataset
        self.replay_buffer = replay_buffer
        self.collect_reference_motions = collect_reference_motions

        '''Define models for AMP'''
        self.policy = ActorAMP(cfg.Model, observation_space, action_space, self.se).to(self.device)
        self.value = Critic(cfg.Model, observation_space, action_space, self.se).to(self.device)
        self.discriminator = Critic(cfg.Model, amp_observation_space, action_space, self.se,
                                    cfg_name='discriminator').to(self.device)
        self.models = {"policy": self.policy, "value": self.value, "discriminator": self.discriminator}
        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value
        self.checkpoint_modules["discriminator"] = self.discriminator
        self.rofunc_logger.module(f"Policy model: {self.policy}")
        self.rofunc_logger.module(f"Value model: {self.value}")
        self.rofunc_logger.module(f"Discriminator model: {self.discriminator}")

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
        self.memory.create_tensor(name="next_states", size=state_tensor_size, dtype=torch.float32, keep_dimensions=kd)

        self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
        self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
        self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="amp_states", size=amp_observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="next_values", size=1, dtype=torch.float32)
        # tensors sampled during training
        self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "log_prob", "values",
                               "returns", "advantages", "amp_states", "next_values"]

        '''Get hyper-parameters from config'''
        self._discount = self.cfg.Agent.discount
        self._td_lambda = self.cfg.Agent.td_lambda
        self._learning_epochs = self.cfg.Agent.learning_epochs
        self._mini_batch_size = self.cfg.Agent.mini_batch_size
        self._amp_batch_size = self.cfg.Agent.amp_batch_size
        self._task_reward_weight = self.cfg.Agent.task_reward_weight
        self._style_reward_weight = self.cfg.Agent.style_reward_weight
        self._lr_a = self.cfg.Agent.lr_a
        self._lr_c = self.cfg.Agent.lr_c
        self._lr_d = self.cfg.Agent.lr_d
        self._lr_scheduler = self.cfg.get("Agent", {}).get("lr_scheduler", KLAdaptiveRL)
        self._lr_scheduler_kwargs = self.cfg.get("Agent", {}).get("lr_scheduler_kwargs", {'kl_threshold': 0.008})
        self._adam_eps = self.cfg.Agent.adam_eps
        self._entropy_loss_scale = self.cfg.Agent.entropy_loss_scale
        self._value_loss_scale = self.cfg.Agent.value_loss_scale
        self._grad_norm_clip = self.cfg.Agent.grad_norm_clip
        self._ratio_clip = self.cfg.Agent.ratio_clip
        self._value_clip = self.cfg.Agent.value_clip
        self._clip_predicted_values = self.cfg.Agent.clip_predicted_values
        self._least_square_discriminator = self.cfg.Agent.least_square_discriminator
        self._discriminator_loss_scale = self.cfg.Agent.discriminator_loss_scale
        self._discriminator_batch_size = self.cfg.Agent.discriminator_batch_size
        self._discriminator_reward_scale = self.cfg.Agent.discriminator_reward_scale
        self._discriminator_logit_regularization_scale = self.cfg.Agent.discriminator_logit_regularization_scale
        self._discriminator_gradient_penalty_scale = self.cfg.Agent.discriminator_gradient_penalty_scale
        self._discriminator_weight_decay_scale = self.cfg.Agent.discriminator_weight_decay_scale
        self._rewards_shaper = self.cfg.get("Agent", {}).get("rewards_shaper", lambda rewards: rewards * 0.01)
        self._state_preprocessor = RunningStandardScaler
        self._state_preprocessor_kwargs = self.cfg.get("Agent", {}).get("state_preprocessor_kwargs",
                                                                        {"size": observation_space, "device": device})
        self._value_preprocessor = RunningStandardScaler
        self._value_preprocessor_kwargs = self.cfg.get("Agent", {}).get("value_preprocessor_kwargs",
                                                                        {"size": 1, "device": device})
        self._amp_state_preprocessor = RunningStandardScaler
        self._amp_state_preprocessor_kwargs = self.cfg.get("Agent", {}).get("amp_state_preprocessor_kwargs",
                                                                            {"size": amp_observation_space,
                                                                             "device": device})

        '''Misc variables'''
        self._current_log_prob = None
        self._current_next_states = None
        self._current_states = None

        self._set_up()

    def _set_up(self):
        """
        Set up optimizer, learning rate scheduler and state/value preprocessors
        """
        assert hasattr(self, "policy"), "Policy is not defined."
        assert hasattr(self, "value"), "Value is not defined."

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

        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self._lr_d, eps=self._adam_eps)
        if self._lr_scheduler is not None:
            self.scheduler_disc = self._lr_scheduler(self.optimizer_disc, **self._lr_scheduler_kwargs)
        self.checkpoint_modules["optimizer_disc"] = self.optimizer_disc

        # set up preprocessors
        super()._set_up()
        if self._amp_state_preprocessor:
            self._amp_state_preprocessor = self._amp_state_preprocessor(**self._amp_state_preprocessor_kwargs)
            self.checkpoint_modules["amp_state_preprocessor"] = self._amp_state_preprocessor
        else:
            self._amp_state_preprocessor = empty_preprocessor

        # Create tensors for motion dataset and replay buffer
        self.motion_dataset.create_tensor(name="states", size=self.amp_observation_space, dtype=torch.float32)
        self.replay_buffer.create_tensor(name="states", size=self.amp_observation_space, dtype=torch.float32)
        
        self._initialize_motion_dataset()

    def _initialize_motion_dataset(self):
        if self.collect_reference_motions is not None:
            for _ in range(math.ceil(self.motion_dataset.memory_size / self._amp_batch_size)):
                self.motion_dataset.add_samples(states=self.collect_reference_motions(self._amp_batch_size))

    def act(self, states: torch.Tensor, deterministic: bool = False):
        if self._current_states is not None:
            states = self._current_states

        res_dict = self.policy(self._state_preprocessor(states), deterministic=deterministic)
        actions = res_dict["action"]
        log_prob = res_dict["log_prob"]
        self._current_log_prob = log_prob
        return actions, log_prob

    def store_transition(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor,
                         rewards: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, infos: torch.Tensor):
        if self._current_states is not None:
            states = self._current_states

        super().store_transition(states=states, actions=actions, next_states=next_states, rewards=rewards,
                                 terminated=terminated, truncated=truncated, infos=infos)

        amp_states = infos["amp_obs"]

        # reward shaping
        if self._rewards_shaper is not None:
            rewards = self._rewards_shaper(rewards)

        # compute values
        values = self.value(self._state_preprocessor(states))
        values = self._value_preprocessor(values, inverse=True)

        next_values = self.value(self._state_preprocessor(next_states))
        next_values = self._value_preprocessor(next_values, inverse=True)
        next_values *= infos['terminate'].view(-1, 1).logical_not()

        # storage transition in memory
        self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                terminated=terminated, truncated=truncated, log_prob=self._current_log_prob,
                                values=values, amp_states=amp_states, next_values=next_values)

    def update_net(self):
        """
        Update the network
        """
        # update dataset of reference motions
        self.motion_dataset.add_samples(states=self.collect_reference_motions(self._amp_batch_size))

        '''Compute combined rewards'''
        rewards = self.memory.get_tensor_by_name("rewards")
        amp_states = self.memory.get_tensor_by_name("amp_states")

        with torch.no_grad():
            amp_logits = self.discriminator(self._amp_state_preprocessor(amp_states))
            if self._least_square_discriminator:
                style_rewards = torch.maximum(torch.tensor(1 - 0.25 * torch.square(1 - amp_logits)),
                                              torch.tensor(0.0001, device=self.device))
            else:
                style_rewards = -torch.log(torch.maximum(torch.tensor(1 - 1 / (1 + torch.exp(-amp_logits))),
                                                         torch.tensor(0.0001, device=self.device)))
            style_rewards *= self._discriminator_reward_scale

        combined_rewards = self._task_reward_weight * rewards + self._style_reward_weight * style_rewards

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
        sampled_motion_batches = self.motion_dataset.sample(names=["states"],
                                                            batch_size=self.memory.memory_size * self.memory.num_envs,
                                                            mini_batches=self._mini_batch_size)

        if len(self.replay_buffer):
            sampled_replay_batches = self.replay_buffer.sample(names=["states"],
                                                               batch_size=self.memory.memory_size * self.memory.num_envs,
                                                               mini_batches=self._mini_batch_size)
        else:
            sampled_replay_batches = [[batches[self._tensors_names.index("amp_states")]] for batches in sampled_batches]

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_discriminator_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            # mini-batches loop
            for i, (sampled_states, sampled_actions, sampled_rewards, samples_next_states, samples_terminated,
                    sampled_log_prob, sampled_values, sampled_returns, sampled_advantages, sampled_amp_states,
                    _) in enumerate(sampled_batches):
                sampled_states = self._state_preprocessor(sampled_states, train=True)
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

                # compute discriminator loss
                if self._discriminator_batch_size:
                    sampled_amp_states_batch = self._amp_state_preprocessor(
                        sampled_amp_states[0:self._discriminator_batch_size], train=True)
                    sampled_amp_replay_states = self._amp_state_preprocessor(
                        sampled_replay_batches[i][0][0:self._discriminator_batch_size], train=True)
                    sampled_amp_motion_states = self._amp_state_preprocessor(
                        sampled_motion_batches[i][0][0:self._discriminator_batch_size], train=True)
                else:
                    sampled_amp_states_batch = self._amp_state_preprocessor(sampled_amp_states, train=True)
                    sampled_amp_replay_states = self._amp_state_preprocessor(sampled_replay_batches[i][0], train=True)
                    sampled_amp_motion_states = self._amp_state_preprocessor(sampled_motion_batches[i][0], train=True)

                sampled_amp_motion_states.requires_grad_(True)
                amp_logits = self.discriminator(sampled_amp_states_batch)
                amp_replay_logits = self.discriminator(sampled_amp_replay_states)
                amp_motion_logits = self.discriminator(sampled_amp_motion_states)
                amp_cat_logits = torch.cat([amp_logits, amp_replay_logits], dim=0)

                # discriminator prediction loss
                if self._least_square_discriminator:
                    discriminator_loss = 0.5 * (
                            F.mse_loss(amp_cat_logits, -torch.ones_like(amp_cat_logits), reduction='mean')
                            + F.mse_loss(amp_motion_logits, torch.ones_like(amp_motion_logits), reduction='mean'))
                else:
                    discriminator_loss = 0.5 * (nn.BCEWithLogitsLoss()(amp_cat_logits, torch.zeros_like(amp_cat_logits))
                                                + nn.BCEWithLogitsLoss()(amp_motion_logits,
                                                                         torch.ones_like(amp_motion_logits)))

                # discriminator logit regularization
                if self._discriminator_logit_regularization_scale:
                    logit_weights = torch.flatten(list(self.discriminator.modules())[-1].weight)
                    discriminator_loss += self._discriminator_logit_regularization_scale * torch.sum(
                        torch.square(logit_weights))

                # discriminator gradient penalty
                if self._discriminator_gradient_penalty_scale:
                    amp_motion_gradient = torch.autograd.grad(amp_motion_logits,
                                                              sampled_amp_motion_states,
                                                              grad_outputs=torch.ones_like(amp_motion_logits),
                                                              create_graph=True,
                                                              retain_graph=True,
                                                              only_inputs=True)
                    gradient_penalty = torch.sum(torch.square(amp_motion_gradient[0]), dim=-1).mean()
                    discriminator_loss += self._discriminator_gradient_penalty_scale * gradient_penalty

                # discriminator weight decay
                if self._discriminator_weight_decay_scale:
                    weights = [torch.flatten(module.weight) for module in self.discriminator.modules()
                               if isinstance(module, torch.nn.Linear)]
                    weight_decay = torch.sum(torch.square(torch.cat(weights, dim=-1)))
                    discriminator_loss += self._discriminator_weight_decay_scale * weight_decay

                discriminator_loss *= self._discriminator_loss_scale

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

                # Update discriminator network
                self.optimizer_disc.zero_grad()
                discriminator_loss.backward()
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.discriminator.parameters(), self._grad_norm_clip)
                self.optimizer_disc.step()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                cumulative_discriminator_loss += discriminator_loss.item()

            # update learning rate
            if self._lr_scheduler:
                self.scheduler_policy.step()
                self.scheduler_value.step()
                self.scheduler_disc.step()

        # update AMP replay buffer
        self.replay_buffer.add_samples(states=amp_states.view(-1, amp_states.shape[-1]))

        # record data
        self.track_data("Info / Combined rewards", combined_rewards.mean().cpu())
        self.track_data("Info / Style rewards", style_rewards.mean().cpu())
        self.track_data("Info / Task rewards", rewards.mean().cpu())

        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batch_size))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batch_size))
        self.track_data("Loss / Discriminator loss",
                        cumulative_discriminator_loss / (self._learning_epochs * self._mini_batch_size))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss",
                            cumulative_entropy_loss / (self._learning_epochs * self._mini_batch_size))
        if self._lr_scheduler:
            self.track_data("Learning / Learning rate (policy)", self.scheduler_policy.get_last_lr()[0])
            self.track_data("Learning / Learning rate (value)", self.scheduler_value.get_last_lr()[0])
            self.track_data("Learning / Learning rate (discriminator)", self.scheduler_disc.get_last_lr()[0])
