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
import torch

from rofunc.learning.RofuncRL.agents.mixline.ase_agent import ASEAgent
from rofunc.learning.RofuncRL.agents.mixline.ase_hrl_agent import ASEHRLAgent
from rofunc.learning.RofuncRL.trainers.base_trainer import BaseTrainer
from rofunc.learning.RofuncRL.utils.memory import RandomMemory


class ASETrainer(BaseTrainer):
    def __init__(self, cfg, env, device, env_name, hrl=False, inference=False):
        super().__init__(cfg, env, device, env_name, inference)
        self.memory = RandomMemory(memory_size=self.rollouts, num_envs=self.env.num_envs, device=device)
        self.motion_dataset = RandomMemory(memory_size=200000, device=device)
        self.replay_buffer = RandomMemory(memory_size=1000000, device=device)
        self.collect_observation = lambda: self.env.reset_done()[0]["obs"]
        self.hrl = hrl
        if self.hrl:
            self.agent = ASEHRLAgent(cfg.train, self.env.observation_space, self.env.action_space, self.memory,
                                     device, self.exp_dir, self.rofunc_logger,
                                     amp_observation_space=self.env.amp_observation_space,
                                     motion_dataset=self.motion_dataset,
                                     replay_buffer=self.replay_buffer,
                                     collect_reference_motions=lambda num_samples: self.env.fetch_amp_obs_demo(
                                         num_samples),
                                     task_related_state_size=self.env.get_task_obs_size())
        else:
            self.agent = ASEAgent(cfg.train, self.env.observation_space, self.env.action_space, self.memory,
                                  device, self.exp_dir, self.rofunc_logger,
                                  amp_observation_space=self.env.amp_observation_space,
                                  motion_dataset=self.motion_dataset,
                                  replay_buffer=self.replay_buffer,
                                  collect_reference_motions=lambda num_samples: self.env.fetch_amp_obs_demo(
                                      num_samples))

        '''Misc variables'''
        self._latent_reset_steps = torch.zeros(self.env.num_envs, dtype=torch.int32).to(self.device)
        self._latent_steps_min = self.cfg.train.Agent.ase_latent_steps_min
        self._latent_steps_max = self.cfg.train.Agent.ase_latent_steps_max

    def _reset_latents(self, env_ids):
        # Equ. 11, provide the model with a latent space
        z_bar = torch.normal(torch.zeros([len(env_ids), self.agent._ase_latent_dim]))
        self.agent._ase_latents[env_ids] = torch.nn.functional.normalize(z_bar, dim=-1).to(self.device)

    def _reset_latent_step_count(self, env_ids):
        self._latent_reset_steps[env_ids] = torch.randint_like(self._latent_reset_steps[env_ids],
                                                               low=self._latent_steps_min,
                                                               high=self._latent_steps_max)

    def _update_latents(self):
        new_latent_envs = self._latent_reset_steps <= self.env.progress_buf.to(self.device)

        need_update = torch.any(new_latent_envs)
        if need_update:
            new_latent_env_ids = new_latent_envs.nonzero(as_tuple=False).flatten()
            self._reset_latents(new_latent_env_ids)
            self._latent_reset_steps[new_latent_env_ids] += torch.randint_like(
                self._latent_reset_steps[new_latent_env_ids],
                low=self._latent_steps_min,
                high=self._latent_steps_max)

    def pre_interaction(self):
        # if self.hrl and self.agent._llc_step == 0:
        #     if self.collect_observation is not None:  # Reset failed envs
        #         self.env.reset_buf = self.env.reset_buf + self.agent.need_reset
        #         obs_dict, done_env_ids = self.env.reset_done()
        #         self.agent._current_states = obs_dict["obs"]
        if self.hrl:
            if self.collect_observation is not None:  # Reset failed envs
                obs_dict, done_env_ids = self.env.reset_done()
                self.agent._current_states = obs_dict["obs"]
        elif not self.hrl:
            if self.collect_observation is not None:  # Reset failed envs
                obs_dict, done_env_ids = self.env.reset_done()
                obs_dict, done_env_ids = self.agent.multi_gpu_transfer(obs_dict, done_env_ids)
                self.agent._current_states = obs_dict["obs"]
                if len(done_env_ids) > 0:
                    self._reset_latents(done_env_ids)
                    self._reset_latent_step_count(done_env_ids)
            self._update_latents()

    def post_interaction(self):
        # if self.agent._llc_step == self.cfg.train.Agent.llc_steps_per_high_action:
        self._rollout += 1
        # self.agent._llc_step = 0

        # Update agent
        if not self._rollout % self.rollouts and self._step >= self.start_learning_steps and self._rollout > 0:
            self.agent.update_net()
            self._update_times += 1
            self.rofunc_logger.info(f'Update {self._update_times} times.', local_verbose=False)

        super().post_interaction()
