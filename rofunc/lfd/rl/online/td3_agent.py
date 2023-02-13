"""
Modified from SKRL
"""

import copy
import itertools
import os.path
from typing import Union, Tuple, Dict, Any

import rofunc as rf
import numpy as np
import gym
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model


class TD3Agent(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Union[Memory, Tuple[Memory], None] = None,
                 observation_space: Union[int, Tuple[int], gym.Space, None] = None,
                 action_space: Union[int, Tuple[int], gym.Space, None] = None,
                 device: Union[str, torch.device] = "cuda:0",
                 cfg: dict = {}) -> None:
        """Twin Delayed DDPG (TD3)

        https://arxiv.org/abs/1802.09477

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: None)
        :type observation_space: int, tuple or list of integers, gym.Space or None, optional
        :param action_space: Action space or shape (default: None)
        :type action_space: int, tuple or list of integers, gym.Space or None, optional
        :param device: Computing device (default: "cuda:0")
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        TD3_DEFAULT_CONFIG = rf.config.omegaconf_to_dict(OmegaConf.load(
            os.path.join(rf.file.get_rofunc_path(), 'config/learning/rl/agent/td3_default_config_skrl.yaml')))
        _cfg = copy.deepcopy(TD3_DEFAULT_CONFIG)
        _cfg.update(cfg)
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

        # models
        self.policy = self.models.get("policy", None)
        self.target_policy = self.models.get("target_policy", None)
        self.critic_1 = self.models.get("critic_1", None)
        self.critic_2 = self.models.get("critic_2", None)
        self.target_critic_1 = self.models.get("target_critic_1", None)
        self.target_critic_2 = self.models.get("target_critic_2", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["target_policy"] = self.target_policy
        self.checkpoint_modules["critic_1"] = self.critic_1
        self.checkpoint_modules["critic_2"] = self.critic_2
        self.checkpoint_modules["target_critic_1"] = self.target_critic_1
        self.checkpoint_modules["target_critic_2"] = self.target_critic_2

        if self.target_policy is not None and self.target_critic_1 is not None and self.target_critic_2 is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_policy.freeze_parameters(True)
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)

            # update target networks (hard update)
            self.target_policy.update_parameters(self.policy, polyak=1)
            self.target_critic_1.update_parameters(self.critic_1, polyak=1)
            self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._exploration_noise = self.cfg["exploration"]["noise"]
        self._exploration_initial_scale = self.cfg["exploration"]["initial_scale"]
        self._exploration_final_scale = self.cfg["exploration"]["final_scale"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._policy_delay = self.cfg["policy_delay"]
        self._critic_update_counter = 0

        self._smooth_regularization_noise = self.cfg["smooth_regularization_noise"]
        self._smooth_regularization_clip = self.cfg["smooth_regularization_clip"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic_1 is not None and self.critic_2 is not None:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._actor_learning_rate)
            self.critic_optimizer = torch.optim.Adam(
                itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                lr=self._critic_learning_rate)
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(self.policy_optimizer,
                                                                      **self.cfg["learning_rate_scheduler_kwargs"])
                self.critic_scheduler = self._learning_rate_scheduler(self.critic_optimizer,
                                                                      **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    def init(self) -> None:
        """Initialize the agent
        """
        super().init()

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="dones", size=1, dtype=torch.bool)

        self.tensors_names = ["states", "actions", "rewards", "next_states", "dones"]

        # clip noise bounds
        self.clip_actions_min = torch.tensor(self.action_space.low, device=self.device)
        self.clip_actions_max = torch.tensor(self.action_space.high, device=self.device)

        # backward compatibility: torch < 1.9 clamp method does not support tensors
        self._backward_compatibility = tuple(map(int, (torch.__version__.split(".")[:2]))) < (1, 9)

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        states = self._state_preprocessor(states)

        # sample random actions
        if timestep < self._random_timesteps:
            return self.policy.random_act(states, taken_actions=None, role="policy")

        # sample deterministic actions
        actions = self.policy.act(states, taken_actions=None, role="policy")

        # add noise
        if self._exploration_noise is not None:
            # sample noises
            noises = self._exploration_noise.sample(actions[0].shape)

            # define exploration timesteps
            scale = self._exploration_final_scale
            if self._exploration_timesteps is None:
                self._exploration_timesteps = timesteps

            # apply exploration noise
            if timestep <= self._exploration_timesteps:
                scale = (1 - timestep / self._exploration_timesteps) \
                        * (self._exploration_initial_scale - self._exploration_final_scale) \
                        + self._exploration_final_scale
                noises.mul_(scale)

                # modify actions
                actions[0].add_(noises)

                if self._backward_compatibility:
                    actions = (torch.max(torch.min(actions[0], self.clip_actions_max), self.clip_actions_min),
                               actions[1],
                               actions[2])
                else:
                    actions[0].clamp_(min=self.clip_actions_min, max=self.clip_actions_max)

                # record noises
                self.track_data("Exploration / Exploration noise (max)", torch.max(noises).item())
                self.track_data("Exploration / Exploration noise (min)", torch.min(noises).item())
                self.track_data("Exploration / Exploration noise (mean)", torch.mean(noises).item())

            else:
                # record noises
                self.track_data("Exploration / Exploration noise (max)", 0)
                self.track_data("Exploration / Exploration noise (min)", 0)
                self.track_data("Exploration / Exploration noise (mean)", 0)

        return actions

    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          dones: torch.Tensor,
                          infos: Any,
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param dones: Signals to indicate that episodes have ended
        :type dones: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(states, actions, rewards, next_states, dones, infos, timestep, timesteps)

        # reward shaping
        if self._rewards_shaper is not None:
            rewards = self._rewards_shaper(rewards, timestep, timesteps)

        if self.memory is not None:
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                    dones=dones)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                   dones=dones)

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts:
            self._update(timestep, timesteps)

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # sample a batch from memory
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = \
            self.memory.sample(names=self.tensors_names, batch_size=self._batch_size)[0]

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            sampled_states = self._state_preprocessor(sampled_states, train=not gradient_step)
            sampled_next_states = self._state_preprocessor(sampled_next_states)

            with torch.no_grad():
                # target policy smoothing
                next_actions, _, _ = self.target_policy.act(states=sampled_next_states, taken_actions=None,
                                                            role="target_policy")
                noises = torch.clamp(self._smooth_regularization_noise.sample(next_actions.shape),
                                     min=-self._smooth_regularization_clip,
                                     max=self._smooth_regularization_clip)
                next_actions.add_(noises)

                if self._backward_compatibility:
                    next_actions = torch.max(torch.min(next_actions, self.clip_actions_max), self.clip_actions_min)
                else:
                    next_actions.clamp_(min=self.clip_actions_min, max=self.clip_actions_max)

                # compute target values
                target_q1_values, _, _ = self.target_critic_1.act(states=sampled_next_states,
                                                                  taken_actions=next_actions, role="target_critic_1")
                target_q2_values, _, _ = self.target_critic_2.act(states=sampled_next_states,
                                                                  taken_actions=next_actions, role="target_critic_2")
                target_q_values = torch.min(target_q1_values, target_q2_values)
                target_values = sampled_rewards + self._discount_factor * sampled_dones.logical_not() * target_q_values

            # compute critic loss
            critic_1_values, _, _ = self.critic_1.act(states=sampled_states, taken_actions=sampled_actions,
                                                      role="critic_1")
            critic_2_values, _, _ = self.critic_2.act(states=sampled_states, taken_actions=sampled_actions,
                                                      role="critic_2")

            critic_loss = F.mse_loss(critic_1_values, target_values) + F.mse_loss(critic_2_values, target_values)

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # delayed update
            self._critic_update_counter += 1
            if not self._critic_update_counter % self._policy_delay:
                # compute policy (actor) loss
                actions, _, _ = self.policy.act(states=sampled_states, taken_actions=None, role="policy")
                critic_values, _, _ = self.critic_1.act(states=sampled_states, taken_actions=actions, role="critic_1")

                policy_loss = -critic_values.mean()

                # optimization step (policy)
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # update target networks
                self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
                self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)
                self.target_policy.update_parameters(self.policy, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

            # record data
            if not self._critic_update_counter % self._policy_delay:
                self.track_data("Loss / Policy loss", policy_loss.item())
            self.track_data("Loss / Critic loss", critic_loss.item())

            self.track_data("Q-network / Q1 (max)", torch.max(critic_1_values).item())
            self.track_data("Q-network / Q1 (min)", torch.min(critic_1_values).item())
            self.track_data("Q-network / Q1 (mean)", torch.mean(critic_1_values).item())

            self.track_data("Q-network / Q2 (max)", torch.max(critic_2_values).item())
            self.track_data("Q-network / Q2 (min)", torch.min(critic_2_values).item())
            self.track_data("Q-network / Q2 (mean)", torch.mean(critic_2_values).item())

            self.track_data("Target / Target (max)", torch.max(target_values).item())
            self.track_data("Target / Target (min)", torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())

            if self._learning_rate_scheduler:
                self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])
