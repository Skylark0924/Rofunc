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

from rofunc.learning.RofuncRL.agents.mixline.amp_agent import AMPAgent
from rofunc.learning.RofuncRL.trainers.base_trainer import BaseTrainer
from rofunc.learning.RofuncRL.utils.memory import RandomMemory


class AMPTrainer(BaseTrainer):
    def __init__(self, cfg, env, device, env_name):
        super().__init__(cfg, env, device, env_name)
        self.memory = RandomMemory(memory_size=self.rollouts, num_envs=self.env.num_envs, device=device)
        self.motion_dataset = RandomMemory(memory_size=200000, device=device)
        self.replay_buffer = RandomMemory(memory_size=1000000, device=device)
        self.collect_observation = lambda: self.env.reset_done()[0]["obs"]
        self.agent = AMPAgent(cfg, self.env.observation_space, self.env.action_space, self.memory,
                              device, self.exp_dir, self.rofunc_logger,
                              amp_observation_space=self.env.amp_observation_space,
                              motion_dataset=self.motion_dataset,
                              replay_buffer=self.replay_buffer,
                              collect_reference_motions=lambda num_samples: self.env.fetch_amp_obs_demo(num_samples))
        self.setup_wandb()

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
