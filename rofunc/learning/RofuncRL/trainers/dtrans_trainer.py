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

import tqdm

from rofunc.learning.RofuncRL.agents.offline.dtrans_agent import DTransAgent
from rofunc.learning.RofuncRL.trainers.base_trainer import BaseTrainer


class DTransTrainer(BaseTrainer):
    def __init__(self, cfg, env, device, env_name):
        super().__init__(cfg, env, device, env_name)
        self.agent = DTransAgent(cfg, self.env.observation_space, self.env.action_space, self.memory,
                                 device, self.exp_dir, self.rofunc_logger)
        self.setup_wandb()

    def train(self):
        """
        Main training loop.
        """
        with tqdm.trange(self.maximum_steps, ncols=80, colour='green') as self.t_bar:
            for _ in self.t_bar:
                self.agent.update_net()

        # close the logger
        self.writer.close()
        self.rofunc_logger.info('Training complete.')
