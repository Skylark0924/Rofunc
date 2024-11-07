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

from rofunc.learning.RofuncRL.agents.online.sac_agent import SACAgent
from rofunc.learning.RofuncRL.trainers.base_trainer import BaseTrainer
from rofunc.learning.RofuncRL.utils.memory import RandomMemory


class SACTrainer(BaseTrainer):
    def __init__(self, cfg, env, device, env_name, **kwargs):
        super().__init__(cfg, env, device, env_name, **kwargs)
        self.memory = RandomMemory(memory_size=cfg.train.Trainer.get("memory_size", 10000), num_envs=self.env.num_envs,
                                   device=device, replacement=True)
        self.agent = SACAgent(cfg.train, self.env.observation_space, self.env.action_space, self.memory,
                              device, self.exp_dir, self.rofunc_logger)

    def post_interaction(self):
        # Update agent
        if self._step >= self.start_learning_steps:
            self.agent.update_net()
            self._update_times += 1
            self.rofunc_logger.info(f'Update {self._update_times} times.', local_verbose=False)

        super().post_interaction()
