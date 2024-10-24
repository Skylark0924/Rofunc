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

import gym
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from typing import Callable, Union, Tuple, Optional

import rofunc as rf
from rofunc.config.utils import get_config
from rofunc.learning.RofuncRL.agents.base_agent import BaseAgent
from rofunc.learning.RofuncRL.agents.mixline.ase_agent import ASEAgent
from rofunc.learning.RofuncRL.models.actor_models import ActorPPO_Beta, ActorPPO_Gaussian
from rofunc.learning.RofuncRL.models.critic_models import Critic
from rofunc.learning.RofuncRL.processors.schedulers import KLAdaptiveRL
from rofunc.learning.RofuncRL.processors.standard_scaler import RunningStandardScaler
from rofunc.learning.RofuncRL.utils.memory import Memory
from rofunc.learning.RofuncRL.utils.memory import RandomMemory
from rofunc.learning.pre_trained_models.download import model_zoo


class ASEHRLAgent(BaseAgent):
    """
    Adversarial Skill Embeddings (ASE) agent for hierarchical reinforcement learning (HRL) using pre-trained low-level controller. \n
    “ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters”. Peng et al. 2022. https://arxiv.org/abs/2205.01906 \n
    Rofunc documentation: https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/ASE.html
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
                 collect_reference_motions: Optional[Callable[[int], torch.Tensor]] = None,
                 task_related_state_size: Optional[int] = None,
                 num_part: Optional[int] = 1):
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
        :param num_part: Number of parts
        """
        """ASE specific parameters"""
        self._ase_latent_dim = cfg.Agent.ase_latent_dim
        self._task_related_state_size = task_related_state_size
        super().__init__(cfg, observation_space, action_space, memory, device, experiment_dir, rofunc_logger)

        '''Define models for ASE HRL agent'''
        if self.cfg.Model.actor.type == "Beta":
            self.policy = ActorPPO_Beta(cfg.Model, observation_space, self._ase_latent_dim * num_part, self.se).to(
                self.device)
        else:
            self.policy = ActorPPO_Gaussian(cfg.Model, observation_space, self._ase_latent_dim * num_part, self.se).to(
                self.device)
        self.value = Critic(cfg.Model, observation_space, self._ase_latent_dim * num_part, self.se).to(self.device)
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
        self.memory.create_tensor(name="amp_states", size=amp_observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="next_values", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="disc_rewards", size=1, dtype=torch.float32)
        # tensors sampled during training
        self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "log_prob", "values",
                               "returns", "advantages", "amp_states", "next_values", "omega_actions", "disc_rewards"]

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
        args_overrides = ["task=HumanoidASEGetupSwordShield", "train=HumanoidASEGetupSwordShieldASERofuncRL"]
        self.llc_config = get_config('./learning/rl', 'config', args=args_overrides)
        if self.cfg.Agent.llc_ckpt_path is None:
            llc_ckpt_path = model_zoo(name="HumanoidASEGetupSwordShield.pth")
        else:
            llc_ckpt_path = self.cfg.Agent.llc_ckpt_path
        llc_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                               shape=(observation_space.shape[0] - self._task_related_state_size,))
        llc_memory = RandomMemory(memory_size=self.memory.memory_size, num_envs=self.memory.num_envs, device=device)
        self.llc_agent = ASEAgent(self.llc_config.train, llc_observation_space, action_space, llc_memory, device,
                                  experiment_dir, rofunc_logger, amp_observation_space,
                                  motion_dataset, replay_buffer, collect_reference_motions)
        self.llc_agent.load_ckpt(llc_ckpt_path)
        # self._build_llc()

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

        self._set_up()

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

    def _build_llc(self):
        from .utils import ase_network_builder
        from .utils import ase_agent
        from .utils import ase_models
        import yaml
        with open(
                "/home/ubuntu/Github/Knowledge-Universe/Robotics/Roadmap-for-robot-science/rofunc/learning/RofuncRL/agents/mixline/utils/ase_humanoid_hrl.yaml",
                'r') as f:
            llc_config = yaml.load(f, Loader=yaml.SafeLoader)
            llc_config_params = llc_config['params']

        llc_checkpoint = "/home/ubuntu/Github/Knowledge-Universe/Robotics/Roadmap-for-robot-science/rofunc/learning/RofuncRL/agents/mixline/utils/ase_llc_reallusion_sword_shield.pth"
        assert (llc_checkpoint != "")

        network_params = llc_config_params['network']
        network_builder = ase_network_builder.ASEBuilder()
        network_builder.load(network_params)

        network = ase_models.ModelASEContinuous(network_builder)
        llc_agent_config = self._build_llc_agent_config(llc_config_params, network)

        self.llc_agent = ase_agent.ASEAgent('llc', llc_agent_config)
        self.llc_agent.restore(llc_checkpoint)
        print("Loaded LLC checkpoint from {:s}".format(llc_checkpoint))
        self.llc_agent.set_eval()

    def _build_llc_agent_config(self, config_params, network):
        from .utils.observer import RLGPUAlgoObserver

        llc_env_info = {'action_space': gym.spaces.Box(-1.0, 1.0, (31,)),
                        'observation_space': gym.spaces.Box(-np.inf, np.inf, (253,)),
                        'amp_observation_space': gym.spaces.Box(-np.inf, np.inf, (1400,))}
        # obs_space = llc_env_info['observation_space']
        # obs_size = obs_space.shape[0]
        # obs_size -= self._task_related_state_size
        # llc_env_info['observation_space'] = gym.spaces.Box(obs_space.low[:obs_size], obs_space.high[:obs_size])

        config = config_params['config']
        config['network'] = network
        config['num_actors'] = 4096
        config['features'] = {'observer': RLGPUAlgoObserver()}
        config['env_info'] = llc_env_info
        return config

    def _get_llc_action(self, states: torch.Tensor, omega_actions: torch.Tensor):
        # get actions from low-level controller
        task_agnostic_states = states[:, :-self._task_related_state_size]
        z = torch.nn.functional.normalize(omega_actions, dim=-1)
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

    def _get_disc_reward(self, amp_states):
        with torch.no_grad():
            amp_logits = self.llc_agent.discriminator(self.llc_agent._amp_state_preprocessor(amp_states))
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
        amp_obs = infos['amp_obs']
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

        amp_states = infos["amp_obs"]

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
        self.memory.add_samples(states=states, actions=actions,
                                rewards=rewards,
                                next_states=next_states,
                                terminated=terminated, truncated=truncated, log_prob=self._current_log_prob,
                                values=values, amp_states=amp_states, next_values=next_values,
                                omega_actions=self._omega_actions_for_llc,
                                disc_rewards=self._get_disc_reward(amp_obs))

    def update_net(self):
        """
        Update the network
        """
        # update dataset of reference motions
        '''Compute combined rewards'''
        rewards = self.memory.get_tensor_by_name("rewards")
        style_rewards = self.memory.get_tensor_by_name("disc_rewards")
        combined_rewards = self._task_reward_weight * rewards + self._style_reward_weight * style_rewards

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

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            # mini-batches loop
            for i, (sampled_states, _, sampled_rewards, samples_next_states, samples_terminated,
                    sampled_log_prob, sampled_values, sampled_returns, sampled_advantages, sampled_amp_states,
                    _, sampled_omega_actions, _) in enumerate(sampled_batches):
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

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._lr_scheduler:
                self.scheduler_policy.step()
                self.scheduler_value.step()

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
