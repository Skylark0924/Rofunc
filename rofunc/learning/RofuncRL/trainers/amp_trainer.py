from rofunc.learning.RofuncRL.agents.online.amp_agent import AMPAgent
from rofunc.learning.RofuncRL.trainers.base_trainer import BaseTrainer
from rofunc.learning.RofuncRL.utils.memory import RandomMemory


class AMPTrainer(BaseTrainer):
    def __init__(self, cfg, env, device):
        super().__init__(cfg, env, device)
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
