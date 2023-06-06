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
from rofunc.learning.rl.models.actor_models import ActorPPO_Beta, ActorPPO_Gaussian
from rofunc.learning.rl.models.critic_models import CriticPPO
from rofunc.learning.rl.utils.memory import Memory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from rofunc.learning.rl.processors.normalizers import empty_preprocessor

from rofunc.learning.rl.utils.skrl_utils import Shared


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
        # if self.cfg.Model.actor.type == "Beta":
        #     self.policy = ActorPPO_Beta(cfg.Model.actor, observation_space, action_space).to(self.device)
        # else:
        #     self.policy = ActorPPO_Gaussian(cfg.Model.actor, observation_space, action_space).to(self.device)
        # self.value = CriticPPO(cfg.Model.critic, observation_space, action_space).to(self.device)
        self.policy = Shared(observation_space, action_space, device).to(device)
        self.value = self.policy.to(device)
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
        self._mini_batch_size = self.cfg.Agent.mini_batch_size
        self._lr_a = self.cfg.Agent.lr_a
        self._lr_c = self.cfg.Agent.lr_c
        self._lr_scheduler = self.cfg.get("Agent", {}).get("lr_scheduler", KLAdaptiveRL)
        self._lr_scheduler_kwargs = self.cfg.get("Agent", {}).get("lr_scheduler_kwargs", {'kl_threshold': 0.008})
        self._adam_eps = self.cfg.Agent.adam_eps
        self._use_gae = self.cfg.Agent.use_gae
        self._entropy_loss_scale = self.cfg.Agent.entropy_loss_scale
        self._value_loss_scale = self.cfg.Agent.value_loss_scale

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

        # set up optimizer and learning rate scheduler
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

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

        # set up preprocessors  TODO: add preprocessors for state and reward
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self._state_preprocessor_kwargs)
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self._value_preprocessor_kwargs)
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = empty_preprocessor

    def act(self, states: torch.Tensor, deterministic: bool = False):
        # states = self._state_preprocessor(states)

        actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")
        self._current_log_prob = log_prob

        # if not deterministic:
        #     # sample stochastic actions
        #     if self.cfg.Model.actor.type == "Beta":
        #         dist = self.policy.get_dist(states)
        #         actions = dist.rsample()  # Sample the action according to the probability distribution
        #         log_prob = dist.log_prob(actions)  # The log probability density of the action
        #     else:
        #         dist = self.policy.get_dist(states)
        #         actions = dist.rsample()  # Sample the action according to the probability distribution
        #         actions = torch.clip(actions, -1, 1)  # [-max,max]
        #         log_prob = dist.log_prob(actions)  # The log probability density of the action
        #     self._current_log_prob = log_prob
        # else:
        #     # choose deterministic actions for evaluation
        #     if self.cfg.Model.actor.type == "Beta":
        #         actions = self.policy.mean(states).detach()
        #         log_prob = None
        #     else:
        #         actions = self.policy(states).detach()
        #         log_prob = None
        return actions, log_prob

    def store_transition(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor,
                         rewards: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, infos: torch.Tensor):
        super().store_transition(states=states, actions=actions, next_states=next_states, rewards=rewards,
                                 terminated=terminated, truncated=truncated, infos=infos)

        if self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards)

            # compute values
            # values = self.value(self._state_preprocessor(states))
            values, _, outputs = self.value.act({"states": self._state_preprocessor(states)}, role="value")
            values = self._value_preprocessor(values, inverse=True)

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
                next_values = values[i + 1] if i < memory_size - 1 else next_values
                advantage = rewards[i] - values[i] + discount_factor * not_dones[i] * (
                        next_values + lambda_coefficient * advantage)
                advantages[i] = advantage
            # returns computation
            values_target = advantages + values
            # advantage normalization
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return values_target, advantages

        # compute returns and advantages
        with torch.no_grad():
            self.value.train(False)
            # next_values = self.value(self._state_preprocessor(self._current_next_states.float()))
            next_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value")
            self.value.train(True)
        next_values = self._value_preprocessor(next_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        values_target, advantages = compute_gae(rewards=self.memory.get_tensor_by_name("rewards"),
                                                dones=self.memory.get_tensor_by_name("terminated"),
                                                values=values,
                                                next_values=next_values,
                                                discount_factor=self._discount,
                                                lambda_coefficient=self._td_lambda)
        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(values_target, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # ---------------------------------
        # sample mini-batches from memory
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
                # dist_now = self.policy.get_dist(sampled_states)
                # dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                # log_prob_now = dist_now.log_prob(sampled_actions)

                _, log_prob_now, _ = self.policy.act({"states": sampled_states, "taken_actions": sampled_actions},
                                                     role="policy")
                # dist_entropy = self.policy.get_entropy(role="policy")

                # compute approximate KL divergence
                with torch.no_grad():
                    ratio = log_prob_now - sampled_log_prob
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)

                # early stopping with KL divergence
                if self._kl_threshold and kl_divergence > self._kl_threshold:
                    break

                # compute entropy loss
                if self._entropy_loss_scale:
                    entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                else:
                    entropy_loss = 0

                # compute policy loss
                ratio = torch.exp(log_prob_now - sampled_log_prob)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip,
                                                                    1.0 + self._ratio_clip)

                # policy_loss = -torch.min(surrogate, surrogate_clipped).mean() - entropy_loss
                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # compute value loss
                # predicted_values = self.value(sampled_states)
                predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(predicted_values - sampled_values,
                                                                   min=-self._value_clip,
                                                                   max=self._value_clip)
                value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

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
                    policy_loss.backward()
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

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._lr_scheduler:
            if self.policy is self.value:
                self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
            else:
                self.track_data("Learning / Learning rate (policy)", self.scheduler_policy.get_last_lr()[0])
                self.track_data("Learning / Learning rate (value)", self.scheduler_value.get_last_lr()[0])
