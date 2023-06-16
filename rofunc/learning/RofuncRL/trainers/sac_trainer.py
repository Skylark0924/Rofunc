from rofunc.learning.RofuncRL.agents.online.sac_agent import SACAgent
from rofunc.learning.RofuncRL.trainers.base_trainer import BaseTrainer
from rofunc.learning.RofuncRL.utils.memory import RandomMemory


class SACTrainer(BaseTrainer):
    def __init__(self, cfg, env, device):
        super().__init__(cfg, env, device)
        self.memory = RandomMemory(memory_size=10000, num_envs=self.env.num_envs, device=device, replacement=True)
        self.agent = SACAgent(cfg, self.env.observation_space, self.env.action_space, self.memory,
                              device, self.exp_dir, self.rofunc_logger)
        self.setup_wandb()

    def post_interaction(self):
        # Update agent
        if self._step >= self.start_learning_steps:
            self.agent.update_net()
            self._update_times += 1
            self.rofunc_logger.info(f'Update {self._update_times} times.', local_verbose=False)

        super().post_interaction()
