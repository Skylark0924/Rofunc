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

from rofunc.learning.RofuncRL.agents.mixline.amp_agent import AMPAgent
from rofunc.learning.RofuncRL.trainers.base_trainer import BaseTrainer
from rofunc.learning.RofuncRL.utils.memory import RandomMemory


class AMPTrainer(BaseTrainer):
    def __init__(self, cfg, env, device, env_name, **kwargs):
        super().__init__(cfg, env, device, env_name, **kwargs)
        self.memory = RandomMemory(memory_size=self.rollouts, num_envs=self.env.num_envs, device=device)
        self.motion_dataset = RandomMemory(memory_size=200000, device=device)
        self.replay_buffer = RandomMemory(memory_size=1000000, device=device)
        self.collect_observation = lambda: self.env.reset_done()[0]["obs"]
        self.agent = AMPAgent(cfg.train, self.env.observation_space, self.env.action_space, self.memory,
                              device, self.exp_dir, self.rofunc_logger,
                              amp_observation_space=self.env.amp_observation_space,
                              motion_dataset=self.motion_dataset,
                              replay_buffer=self.replay_buffer,
                              collect_reference_motions=lambda num_samples: self.env.fetch_amp_obs_demo(num_samples))

    def pre_interaction(self):
        if self.collect_observation is not None:
            self.agent._current_states = self.collect_observation()

    def post_interaction(self):
        self._rollout += 1

        # Update agent
        if not self._rollout % self.rollouts and self._step >= self.start_learning_steps:
            self.agent.update_net()
            self._update_times += 1
            self.rofunc_logger.info(f'Update {self._update_times} times.', local_verbose=False)

        super().post_interaction()
