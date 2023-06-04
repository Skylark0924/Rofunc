import itertools
from typing import Union, Tuple, Optional

import gym
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

import rofunc as rf
from rofunc.learning.rl.agents.base_agent import BaseAgent
from rofunc.learning.rl.models.actor_models import ActorPPO
from rofunc.learning.rl.models.critic_models import CriticPPO
from rofunc.learning.rl.utils.memory import Memory


class PPOAgent(BaseAgent):
    def __init__(self,
                 cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 experiment_dir: Optional[str] = None,
                 rofunc_logger: Optional[rf.utils.BeautyLogger] = None):
        """
        PPO algorithm. “Proximal Policy Optimization Algorithms”. John Schulman. et al. https://arxiv.org/abs/1707.06347
        :param cfg: Custom configuration
        :param observation_space: Observation/state space or shape
        :param action_space: Action space or shape
        :param memory: Memory for storing transitions
        :param device: Device on which the torch tensor is allocated
        :param experiment_dir: Directory where experiment outputs are saved
        :param rofunc_logger: Rofunc logger
        """
        super().__init__(cfg, observation_space, action_space, memory, device, experiment_dir, rofunc_logger)

        # Define models for PPO
        self.policy = ActorPPO(cfg.Model.actor, observation_space, action_space).to(self.device)
        self.value = CriticPPO(cfg.Model.critic, observation_space, action_space).to(self.device)
        self.models = {"policy": self.policy, "value": self.value}

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            # tensors sampled during training
            self._tensors_names = ["states", "actions", "terminated", "log_prob", "values", "returns", "advantages"]

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # get hyper-parameters
        self._horizon = self.cfg.Agent.horizon
        self._discount = self.cfg.Agent.discount
        self._td_lambda = self.cfg.Agent.td_lambda
        self._learning_epochs = self.cfg.Agent.learning_epochs
        self._batch_size = self.cfg.Agent.batch_size
        self._lr = self.cfg.Agent.lr
        self._lr_scheduler = self.cfg.Agent.lr_scheduler
        self._lr_scheduler_kwargs = self.cfg.Agent.lr_scheduler_kwargs
        self._use_gae = self.cfg.Agent.use_gae
        self._entropy_loss_scale = self.cfg.Agent.entropy_loss_scale
        self._value_loss_scale = self.cfg.Agent.value_loss_scale

        self._grad_norm_clip = self.cfg.Agent.grad_norm_clip
        self._ratio_clip = self.cfg.Agent.ratio_clip
        self._value_clip = self.cfg.Agent.value_clip
        self._clip_predicted_values = self.cfg.Agent.clip_predicted_values
        self._kl_threshold = self.cfg.Agent.kl_threshold

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._lr)
            else:
                self.optimizer = torch.optim.Adam(itertools.chain(self.policy.parameters(), self.value.parameters()),
                                                  lr=self._lr)
            if self._lr_scheduler is not None:
                self.scheduler = self._lr_scheduler(self.optimizer, **self._lr_scheduler_kwargs)

            self.checkpoint_modules["optimizer"] = self.optimizer

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

        # # set up preprocessors  TODO: add preprocessors for state and reward
        # if self._state_preprocessor:
        #     self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
        #     self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        # else:
        #     self._state_preprocessor = self._empty_preprocessor
        #
        # if self._value_preprocessor:
        #     self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
        #     self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        # else:
        #     self._value_preprocessor = self._empty_preprocessor

    def act(self, states: torch.Tensor, timestep: int = None):
        """
        Choose action based on the current state
        :param states: current state
        :param timestep: current timestep
        :return:
        """
        # sample stochastic actions
        actions, log_prob = self.policy.act(states)
        self._current_log_prob = log_prob.reshape((-1, 1))

        return actions, log_prob

    def store_transition(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor,
                         rewards: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, infos: torch.Tensor):
        super().store_transition(states, actions, next_states, rewards, terminated, truncated, infos)

        if self.memory is not None:
            self._current_next_states = next_states

            # compute values
            values = self.value(states)

            # storage transition in memory
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                    terminated=terminated, truncated=truncated, log_prob=self._current_log_prob,
                                    values=values)

    def update_net(self):
        def compute_gae(rewards: torch.Tensor,
                        dones: torch.Tensor,
                        values: torch.Tensor,
                        next_values: torch.Tensor,
                        discount_factor: float = 0.99,
                        lambda_coefficient: float = 0.95):
            """
            Compute the Generalized Advantage Estimator (GAE)
            :param rewards: Rewards obtained by the agent
            :param dones: Signals to indicate that episodes have ended
            :param values: Values obtained by the agent
            :param next_values: Next values obtained by the agent
            :param discount_factor: Discount factor
            :param lambda_coefficient: Lambda coefficient
            :return: Generalized Advantage Estimator
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = rewards[i] - values[i] + discount_factor * not_dones[i] * (
                        next_values + lambda_coefficient * advantage)
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad():
            self.value.train(False)
            last_values = self.value(self._current_next_states.float())
            self.value.train(True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(rewards=self.memory.get_tensor_by_name("rewards"),
                                          dones=self.memory.get_tensor_by_name("terminated"),
                                          values=values,
                                          next_values=last_values,
                                          discount_factor=self._discount,
                                          lambda_coefficient=self._td_lambda)
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._batch_size)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for i, (sampled_states, sampled_actions, sampled_dones, sampled_log_prob, sampled_values, sampled_returns,
                    sampled_advantages) in enumerate(sampled_batches):
                # sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                next_log_prob, entropy = self.policy.get_log_prob_entropy(sampled_states, sampled_actions)

                # compute approximate KL divergence
                with torch.no_grad():
                    ratio = next_log_prob - sampled_log_prob
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)

                # early stopping with KL divergence
                if self._kl_threshold and kl_divergence > self._kl_threshold:
                    break

                # compute entropy loss
                if self._entropy_loss_scale:
                    entropy_loss = -self._entropy_loss_scale * entropy.mean()
                else:
                    entropy_loss = 0

                # compute policy loss
                ratio = torch.exp(next_log_prob - sampled_log_prob)
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

                # optimization step
                self.optimizer.zero_grad()
                (policy_loss + entropy_loss + value_loss).backward()
                if self._grad_norm_clip > 0:
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(itertools.chain(self.policy.parameters(), self.value.parameters()),
                                                 self._grad_norm_clip)
                self.optimizer.step()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # # update learning rate
            # if self._lr_scheduler:
            #     if isinstance(self.scheduler, KLAdaptiveRL):
            #         self.scheduler.step(torch.tensor(kl_divergences).mean())
            #     else:
            #         self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._batch_size))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._batch_size))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._batch_size))

        # self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._lr_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
