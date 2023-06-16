from rofunc.learning.RofuncRL.agents.online.a2c_agent import A2CAgent
from rofunc.learning.RofuncRL.trainers.base_trainer import BaseTrainer
from rofunc.learning.RofuncRL.utils.memory import RandomMemory


class A2CTrainer(BaseTrainer):
    def __init__(self, cfg, env, device):
        super().__init__(cfg, env, device)
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
