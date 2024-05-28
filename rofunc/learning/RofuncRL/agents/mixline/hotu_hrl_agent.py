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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

import rofunc as rf
from rofunc.config.utils import get_config
from rofunc.learning.RofuncRL.agents.base_agent import BaseAgent
from rofunc.learning.RofuncRL.agents.mixline.hotu_llc_agent import HOTULLCAgent
from rofunc.learning.RofuncRL.models.actor_models import ActorPPO_Beta, ActorPPO_Gaussian
from rofunc.learning.RofuncRL.models.critic_models import Critic
from rofunc.learning.RofuncRL.processors.schedulers import KLAdaptiveRL
from rofunc.learning.RofuncRL.processors.standard_scaler import RunningStandardScaler
from rofunc.learning.RofuncRL.processors.standard_scaler import empty_preprocessor
from rofunc.learning.RofuncRL.utils.memory import Memory
from rofunc.learning.RofuncRL.utils.memory import RandomMemory
from rofunc.learning.pre_trained_models.download import model_zoo


class HOTUHRLAgent(BaseAgent):
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
                 collect_reference_motions: Optional[Callable[[int], torch.Tensor]] = None,
                 task_related_state_size: Optional[int] = None, ):
        """
        :param cfg: Configuration
        :param observation_space: Observation space
        :param action_space: Action space
        :param memory: Memory for storing transitions
        :param device: Device on which the torch tensor is allocated
        :param experiment_dir: Directory where experiment outputs are saved
        :param rofunc_logger: Rofunc logger
        :param amp_observation_space: cfg["env"]["numASEObsSteps"] * NUM_ASE_OBS_PER_STEP
        :param motion_dataset: Motion dataset
        :param replay_buffer: Replay buffer
        :param collect_reference_motions: Function for collecting reference motions
        :param task_related_state_size: Size of task-related states
        """
        self.amp_observation_space = amp_observation_space
        self.motion_dataset = motion_dataset
        self.replay_buffer = replay_buffer
        self.collect_reference_motions = collect_reference_motions

        """ASE specific parameters"""
        self._ase_latent_dim = cfg.Agent.ase_latent_dim
        self._task_related_state_size = task_related_state_size
        super().__init__(cfg, observation_space, action_space, memory, device, experiment_dir, rofunc_logger)

        """HOTU specific parameters"""
        if not isinstance(amp_observation_space, list):
            amp_observation_space = [amp_observation_space]
        self.num_parts = len(amp_observation_space)
        self.whole_amp_obs_space = amp_observation_space  # Whole AMP observation space
        self.motion_dataset_list = []
        self.replay_buffer_list = []
        for i in range(self.num_parts):
            self.motion_dataset_list.append(deepcopy(motion_dataset))
            self.replay_buffer_list.append(deepcopy(replay_buffer))

        super().__init__(cfg, observation_space, action_space, memory, device, experiment_dir, rofunc_logger)

        '''Define models for HOTU HRL agent'''
        if self.cfg.Model.actor.type == "Beta":
            self.policy = ActorPPO_Beta(cfg.Model, observation_space, self._ase_latent_dim * self.num_parts,
                                        self.se).to(self.device)
        else:
            self.policy = ActorPPO_Gaussian(cfg.Model, observation_space, self._ase_latent_dim * self.num_parts,
                                            self.se).to(self.device)
        self.value = Critic(cfg.Model, observation_space, self._ase_latent_dim * self.num_parts, self.se).to(
            self.device)

        self.learn_style = self.cfg.Agent.learn_style

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
        self.memory.create_tensor(name="next_states", size=state_tensor_size, dtype=torch.float32, keep_dimensions=kd)
        self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
        self.memory.create_tensor(name="omega_actions", size=self._ase_latent_dim, dtype=torch.float32)
        self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
        self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="next_values", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="disc_rewards", size=1, dtype=torch.float32)
        self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "log_prob", "values",
                               "returns", "advantages", "next_values", "omega_actions", "disc_rewards"]

        for i in range(self.num_parts):
            self._tensors_names.append(f"amp_states_{i}")
            # self._tensors_names.append(f"disc_rewards_{i}")
            self.memory.create_tensor(name=f"amp_states_{i}", size=amp_observation_space[i], dtype=torch.float32)
            # self.memory.create_tensor(name=f"disc_rewards_{i}", size=1, dtype=torch.float32)

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
        self._value_loss_scale = self.cfg.Agent.value_loss_scale
        self._grad_norm_clip = self.cfg.Agent.grad_norm_clip
        self._ratio_clip = self.cfg.Agent.ratio_clip
        self._value_clip = self.cfg.Agent.value_clip
        self._clip_predicted_values = self.cfg.Agent.clip_predicted_values
        self._task_reward_weight = self.cfg.Agent.task_reward_weight
        self._style_reward_weight = self.cfg.Agent.style_reward_weight
        self._kl_threshold = self.cfg.Agent.kl_threshold
        self._rewards_shaper = None
        # self._rewards_shaper = self.cfg.get("Agent", {}).get("rewards_shaper", lambda rewards: rewards * 0.01)
        self._state_preprocessor = RunningStandardScaler
        self._state_preprocessor_kwargs = self.cfg.get("Agent", {}).get("state_preprocessor_kwargs",
                                                                        {"size": observation_space, "device": device})
        self._value_preprocessor = RunningStandardScaler
        self._value_preprocessor_kwargs = self.cfg.get("Agent", {}).get("value_preprocessor_kwargs",
                                                                        {"size": 1, "device": device})

        """Define pre-trained low-level controller"""
        GlobalHydra.instance().clear()
        args_overrides = ["task=HumanoidHOTUGetup", "train=HumanoidHOTUGetupRofuncRL"]
        self.llc_config = get_config('./learning/rl', 'config', args=args_overrides)
        if self.cfg.Agent.llc_ckpt_path is None:
            llc_ckpt_path = model_zoo(name="HumanoidASEGetupSwordShield.pth")  # TODO
        else:
            llc_ckpt_path = self.cfg.Agent.llc_ckpt_path
        llc_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                               shape=(observation_space.shape[0] - self._task_related_state_size,))
        llc_memory = RandomMemory(memory_size=self.memory.memory_size, num_envs=self.memory.num_envs, device=device)
        self.llc_agent = HOTULLCAgent(self.llc_config.train, llc_observation_space, action_space, llc_memory, device,
                                      experiment_dir, rofunc_logger, amp_observation_space,
                                      motion_dataset, replay_buffer, collect_reference_motions)
        self.llc_agent.load_ckpt(llc_ckpt_path)

        '''Misc variables'''
        self._current_states = None
        self._current_log_prob = None
        self._current_next_states = None
        self._llc_step = 0
        self._omega_actions_for_llc = None
        self.pre_states = None
        self.llc_cum_rew = torch.zeros((self.memory.num_envs, 1), dtype=torch.float32).to(self.device)
        self.llc_cum_disc_rew = torch.zeros((self.memory.num_envs, 1), dtype=torch.float32).to(self.device)
        self.need_reset = torch.zeros((self.memory.num_envs, 1), dtype=torch.float32).to(self.device)
        self.need_terminate = torch.zeros((self.memory.num_envs, 1), dtype=torch.float32).to(self.device)

        if self.learn_style:
            self._build_decompose_model()

        self._set_up()

    def _build_decompose_model(self):
        self._lr_d = self.cfg.Agent.lr_d
        self.hrl_discriminator = Critic(self.cfg.Model, self.whole_amp_obs_space[0], self.action_space, self.se,
                                        cfg_name='discriminator').to(self.device)

        self._amp_state_preprocessor = RunningStandardScaler
        self._amp_state_preprocessor_kwargs = self.cfg.get("Agent", {}).get("amp_state_preprocessor_kwargs",
                                                                            {"size": self.whole_amp_obs_space[0],
                                                                             "device": self.device})
        self.hrl_optimizer_disc = torch.optim.Adam(self.hrl_discriminator.parameters(), lr=self._lr_d,
                                                   eps=self._adam_eps)
        if self._lr_scheduler is not None:
            self.hrl_scheduler_disc = self._lr_scheduler(self.hrl_optimizer_disc, **self._lr_scheduler_kwargs)
        self.checkpoint_modules["hrl_optimizer_disc"] = self.hrl_optimizer_disc

        self.hrl_discriminator_list = [self.hrl_discriminator]
        self._amp_state_preprocessor_kwargs_list = [self._amp_state_preprocessor_kwargs]
        self.hrl_optimizer_disc_list = [self.hrl_optimizer_disc]
        self.hrl_scheduler_disc_list = [self.hrl_scheduler_disc]

        if self._amp_state_preprocessor:
            self._amp_state_preprocessor = self._amp_state_preprocessor(**self._amp_state_preprocessor_kwargs)
            self.checkpoint_modules["amp_state_preprocessor"] = self._amp_state_preprocessor
        else:
            self._amp_state_preprocessor = empty_preprocessor
        self._amp_state_preprocessor_list = [self._amp_state_preprocessor]

        for i in range(1, self.num_parts):
            disc_model = Critic(cfg=self.cfg.Model,
                                observation_space=self.whole_amp_obs_space[i],
                                action_space=self.action_space,
                                state_encoder=self.se,
                                cfg_name='discriminator').to(self.device)
            self.rofunc_logger.module(f"Discriminator model {i}: {disc_model}")

            amp_state_pre_kwargs = {"size": self.whole_amp_obs_space[i], "device": self.device}
            optimizer_disc = torch.optim.Adam(disc_model.parameters(), lr=self._lr_d, eps=self._adam_eps)
            amp_state_preprocessor = RunningStandardScaler(**amp_state_pre_kwargs)

            if self._lr_scheduler is not None:
                hrl_scheduler_disc = self._lr_scheduler(optimizer_disc, **self._lr_scheduler_kwargs)
                self.hrl_scheduler_disc_list.append(hrl_scheduler_disc)

            self.hrl_discriminator_list.append(disc_model)
            self._amp_state_preprocessor_kwargs_list.append(amp_state_pre_kwargs)
            self.hrl_optimizer_disc_list.append(optimizer_disc)
            self._amp_state_preprocessor_list.append(amp_state_preprocessor)

    def _initialize_motion_dataset(self):
        if self.collect_reference_motions is not None:
            for i in range(self.num_parts):
                for _ in range(math.ceil(self.motion_dataset.memory_size / self.llc_agent._amp_batch_size)):
                    self.motion_dataset_list[i].add_samples(
                        states=self.collect_reference_motions(self.llc_agent._amp_batch_size)[i])

    def _set_up(self):
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

        super()._set_up()

        for i in range(self.num_parts):
            self.motion_dataset_list[i].create_tensor(name="states", size=self.whole_amp_obs_space[i],
                                                      dtype=torch.float32)
            self.replay_buffer_list[i].create_tensor(name="states", size=self.whole_amp_obs_space[i],
                                                     dtype=torch.float32)
        self._initialize_motion_dataset()

    def _get_llc_action(self, states: torch.Tensor, omega_actions: torch.Tensor):
        # get actions from low-level controller
        if self._task_related_state_size == 0:
            task_agnostic_states = states
        else:
            task_agnostic_states = states[:, :-self._task_related_state_size]

        omega_actions = omega_actions.view((omega_actions.shape[0], self.num_parts, self._ase_latent_dim))
        z = torch.nn.functional.normalize(omega_actions, dim=-1)
        z = z.view((z.shape[0], self.num_parts * self._ase_latent_dim))

        actions, _ = self.llc_agent.act(task_agnostic_states, deterministic=False, ase_latents=z)
        # actions, _ = self.llc_agent.model.a2c_network.eval_actor(obs=task_agnostic_states, ase_latents=z)
        # self._llc_step += 1
        return actions

    def act(self, states: torch.Tensor, deterministic: bool = False):
        # if self._llc_step == 0:
        if self._current_states is not None:
            states = self._current_states
        self.pre_states = states
        self.llc_cum_rew = torch.zeros((self.memory.num_envs, 1), dtype=torch.float32).to(self.device)
        self.llc_cum_disc_rew = torch.zeros((self.memory.num_envs, 1), dtype=torch.float32).to(self.device)
        self.need_reset = torch.zeros((self.memory.num_envs, 1), dtype=torch.float32).to(self.device)
        self.need_terminate = torch.zeros((self.memory.num_envs, 1), dtype=torch.float32).to(self.device)
        res_dict = self.policy(self._state_preprocessor(states), deterministic=deterministic)
        omega_actions, self._current_log_prob = res_dict["action"], res_dict["log_prob"]

        self._omega_actions_for_llc = omega_actions
        actions = self._get_llc_action(states, self._omega_actions_for_llc)
        return actions, self._current_log_prob

    def _get_disc_reward(self, amp_states, part_i):
        with torch.no_grad():
            amp_logits = self.llc_agent.discriminator_list[part_i](
                self.llc_agent._amp_state_preprocessor_list[part_i](amp_states))
            if self.llc_agent._least_square_discriminator:
                style_rewards = torch.maximum(torch.tensor(1 - 0.25 * torch.square(1 - amp_logits)),
                                              torch.tensor(0.0001, device=self.device))
            else:
                style_rewards = -torch.log(torch.maximum(torch.tensor(1 - 1 / (1 + torch.exp(-amp_logits))),
                                                         torch.tensor(0.0001, device=self.device)))
            style_rewards *= self.llc_agent._discriminator_reward_scale
        return style_rewards

    def store_transition(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor,
                         rewards: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, infos: torch.Tensor):

        # self.llc_cum_rew.add_(rewards)
        # amp_obs = infos['amp_obs']
        # curr_disc_reward = self._get_disc_reward(amp_obs)
        # self.llc_cum_disc_rew.add_(curr_disc_reward)
        # self.need_reset.add_(terminated + truncated)
        # self.need_terminate.add_(infos['terminate'].view(-1, 1))
        # if self._llc_step == self.cfg.Agent.llc_steps_per_high_action:
        # super().store_transition(states=self.pre_states, actions=actions, next_states=next_states,
        #                          rewards=self.llc_cum_rew,
        #                          terminated=self.need_reset, truncated=self.need_reset, infos=infos)
        super().store_transition(states=states, actions=actions, next_states=next_states,
                                 rewards=rewards, terminated=terminated, truncated=truncated, infos=infos)

        # reward shaping
        if self._rewards_shaper is not None:
            rewards = self._rewards_shaper(rewards)

        # compute values
        # values = self.value(self._state_preprocessor(self.pre_states))
        values = self.value(self._state_preprocessor(states))
        values = self._value_preprocessor(values, inverse=True)
        if (values.isnan() == True).any():
            print("values is nan")

        next_values = self.value(self._state_preprocessor(next_states))
        next_values = self._value_preprocessor(next_values, inverse=True)
        next_values *= self.need_terminate.logical_not()

        # storage transition in memory
        # self.memory.add_samples(states=self.pre_states, actions=actions,
        #                         rewards=self.llc_cum_rew,
        #                         next_states=next_states,
        #                         terminated=self.need_reset, truncated=self.need_reset,
        #                         log_prob=self._current_log_prob,
        #                         values=values, amp_states=amp_states, next_values=next_values,
        #                         omega_actions=self._omega_actions_for_llc,
        #                         disc_rewards=self.llc_cum_disc_rew)
        amp_states_dict = {f"amp_states_{i}": infos[f"amp_obs_{i}"] for i in range(self.num_parts)}

        disc_rewards = 1
        for i in range(self.num_parts):
            tmp_disc_rew = self._get_disc_reward(amp_states_dict[f"amp_states_{i}"], i)
            disc_rewards *= tmp_disc_rew
        self.memory.add_samples(states=states, actions=actions,
                                rewards=rewards,
                                next_states=next_states,
                                terminated=terminated, truncated=truncated, log_prob=self._current_log_prob,
                                values=values, next_values=next_values,
                                omega_actions=self._omega_actions_for_llc,
                                disc_rewards=disc_rewards,
                                **amp_states_dict)

    def update_net(self):
        """
        Update the network
        """
        # update dataset of reference motions
        '''Compute combined rewards'''
        rewards = self.memory.get_tensor_by_name("rewards")
        style_rewards = self.memory.get_tensor_by_name("disc_rewards")
        combined_rewards = self._task_reward_weight * rewards + self._style_reward_weight * style_rewards

        if self.learn_style:
            amp_logits_list = []
            for i in range(self.num_parts):
                # update dataset of reference motions
                self.motion_dataset_list[i].add_samples(
                    states=self.llc_agent.collect_reference_motions(self.llc_agent._amp_batch_size)[i])
                amp_logits_list.append(
                    self.hrl_discriminator_list[i](
                        self._amp_state_preprocessor_list[i](self.memory.get_tensor_by_name(f"amp_states_{i}"))))

            hrl_style_rewards = 1
            for i in range(self.num_parts):
                if self.llc_agent._least_square_discriminator:
                    hrl_style_rewards_tmp = torch.maximum(
                        torch.tensor(1 - 0.25 * torch.square(1 - amp_logits_list[i])),
                        torch.tensor(0.0001, device=self.device))
                else:
                    hrl_style_rewards_tmp = -torch.log(
                        torch.maximum(torch.tensor(1 - 1 / (1 + torch.exp(-amp_logits_list[i]))),
                                      torch.tensor(0.0001, device=self.device)))
                hrl_style_rewards *= hrl_style_rewards_tmp
            hrl_style_rewards *= self.llc_agent._discriminator_reward_scale

            combined_rewards += self._style_reward_weight * hrl_style_rewards

        '''Compute Generalized Advantage Estimator (GAE)'''
        values = self.memory.get_tensor_by_name("values")
        next_values = self.memory.get_tensor_by_name("next_values")
        if (values.isnan() == True).any():
            print("values is nan")

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

        # learning epochs
        for epoch in range(self._learning_epochs):
            # mini-batches loop
            for i, sampled_tensors in enumerate(sampled_batches):
                (sampled_states, _, sampled_rewards, samples_next_states, samples_terminated,
                 sampled_log_prob, sampled_values, sampled_returns, sampled_advantages,
                 _, sampled_omega_actions, _) = sampled_tensors[:12]

                sampled_amp_states_list = []
                for part_i in range(self.num_parts):
                    sampled_amp_states = sampled_tensors[self._tensors_names.index(f"amp_states_{part_i}")]
                    sampled_amp_states_list.append(sampled_amp_states)
                assert len(sampled_amp_states_list) == self.num_parts

                sampled_states = self._state_preprocessor(sampled_states, train=True)
                res_dict = self.policy(sampled_states, sampled_omega_actions)
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

                if self.learn_style:
                    discriminator_loss_list = []
                    for part_i in range(self.num_parts):
                        discriminator_loss = self._calc_disc_loss(sampled_amp_states_list[part_i],
                                                                  sampled_replay_batches_list[part_i],
                                                                  sampled_motion_batches_list[part_i],
                                                                  self.hrl_discriminator_list[part_i],
                                                                  self._amp_state_preprocessor_list[part_i],
                                                                  i)
                        discriminator_loss_list.append(discriminator_loss)

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

                if self.learn_style:
                    for part_i in range(self.num_parts):
                        self._update_disc(discriminator_loss_list[part_i], self.hrl_discriminator_list[part_i],
                                          self.hrl_optimizer_disc_list[part_i])

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._lr_scheduler:
                self.scheduler_policy.step()
                self.scheduler_value.step()

                if self.learn_style:
                    for part_i in range(self.num_parts):
                        self.hrl_scheduler_disc_list[part_i].step()

        # record data
        self.track_data("Info / Combined rewards", combined_rewards.mean().cpu())
        self.track_data("Info / Style rewards", style_rewards.mean().cpu())
        self.track_data("Info / Task rewards", rewards.mean().cpu())

        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batch_size))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batch_size))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss",
                            cumulative_entropy_loss / (self._learning_epochs * self._mini_batch_size))
        if self._lr_scheduler:
            self.track_data("Learning / Learning rate (policy)", self.scheduler_policy.get_last_lr()[0])
            self.track_data("Learning / Learning rate (value)", self.scheduler_value.get_last_lr()[0])

    def _calc_disc_loss(self, sampled_amp_states, sampled_replay_batches, sampled_motion_batches,
                        discriminator, amp_state_preprocessor, i):
        # compute discriminator loss
        if self.llc_agent._discriminator_batch_size:
            sampled_amp_states_batch = amp_state_preprocessor(
                sampled_amp_states[0:self.llc_agent._discriminator_batch_size], train=True)
            sampled_amp_replay_states = amp_state_preprocessor(
                sampled_replay_batches[i][0][0:self.llc_agent._discriminator_batch_size], train=True)
            sampled_amp_motion_states = amp_state_preprocessor(
                sampled_motion_batches[i][0][0:self.llc_agent._discriminator_batch_size], train=True)
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
        if self.llc_agent._least_square_discriminator:
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
        if self.llc_agent._discriminator_logit_regularization_scale:
            logit_weights = torch.flatten(list(discriminator.modules())[-1].weight)
            discriminator_loss += self.llc_agent._discriminator_logit_regularization_scale * torch.sum(
                torch.square(logit_weights))

        # discriminator gradient penalty TODO: check whether this is used
        if self.llc_agent._discriminator_gradient_penalty_scale:
            amp_motion_gradient = torch.autograd.grad(amp_motion_logits,
                                                      sampled_amp_motion_states,
                                                      grad_outputs=torch.ones_like(
                                                          amp_motion_logits),
                                                      create_graph=True,
                                                      retain_graph=True,
                                                      only_inputs=True)
            gradient_penalty = torch.sum(torch.square(amp_motion_gradient[0]), dim=-1).mean()
            discriminator_loss += self.llc_agent._discriminator_gradient_penalty_scale * gradient_penalty

        # discriminator weight decay
        if self.llc_agent._discriminator_weight_decay_scale:
            weights = [torch.flatten(module.weight) for module in discriminator.modules()
                       if isinstance(module, torch.nn.Linear)]
            weight_decay = torch.sum(torch.square(torch.cat(weights, dim=-1)))
            discriminator_loss += self.llc_agent._discriminator_weight_decay_scale * weight_decay

        discriminator_loss *= self.llc_agent._discriminator_loss_scale

        return discriminator_loss

    def _update_disc(self, discriminator_loss, discriminator, optimizer_disc):
        # Update discriminator network
        optimizer_disc.zero_grad()
        discriminator_loss.backward()
        if self._grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(), self._grad_norm_clip)
        optimizer_disc.step()
