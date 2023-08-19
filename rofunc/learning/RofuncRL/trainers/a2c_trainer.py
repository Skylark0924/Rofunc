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

from rofunc.learning.RofuncRL.agents.online.a2c_agent import A2CAgent
from rofunc.learning.RofuncRL.trainers.base_trainer import BaseTrainer
from rofunc.learning.RofuncRL.utils.memory import RandomMemory


class A2CTrainer(BaseTrainer):
    def __init__(self, cfg, env, device, env_name):
        super().__init__(cfg, env, device, env_name)
        self.memory = RandomMemory(memory_size=cfg.Trainer.rollouts, num_envs=self.env.num_envs, device=device)
        self.agent = A2CAgent(cfg, self.env.observation_space, self.env.action_space, self.memory,
                              device, self.exp_dir, self.rofunc_logger)
        self.setup_wandb()

    def post_interaction(self):
        self._rollout += 1

        # Update agent
        if not self._rollout % self.rollouts and self._step >= self.start_learning_steps:
            self.agent.update_net()
            self._update_times += 1
            self.rofunc_logger.info(f'Update {self._update_times} times.', local_verbose=False)

        super().post_interaction()
