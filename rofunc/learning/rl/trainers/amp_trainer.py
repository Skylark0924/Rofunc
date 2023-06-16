from rofunc.learning.rl.agents.online.amp_agent import AMPAgent
from rofunc.learning.rl.trainers.base_trainer import BaseTrainer
from rofunc.learning.rl.utils.memory import RandomMemory


class AMPTrainer(BaseTrainer):
    def __init__(self, cfg, env, device):
        super().__init__(cfg, env, device)
        self.memory = RandomMemory(memory_size=cfg.Trainer.rollouts, num_envs=self.env.num_envs, device=device)
        self.agent = AMPAgent(cfg, env.observation_space, env.action_space, self.memory,
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
