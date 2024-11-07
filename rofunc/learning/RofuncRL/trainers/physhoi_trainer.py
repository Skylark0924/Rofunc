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
import tqdm

from rofunc.learning.RofuncRL.agents.mixline.physhoi_agent import PhysHOIAgent
from rofunc.learning.RofuncRL.trainers.base_trainer import BaseTrainer
from rofunc.learning.RofuncRL.utils.memory import RandomMemory


class PhysHOITrainer(BaseTrainer):
    def __init__(self, cfg, env, device, env_name, **kwargs):
        super().__init__(cfg, env, device, env_name, **kwargs)
        self.collect_observation = lambda: self.env.reset_done()[0]["obs"]
        self.memory = RandomMemory(memory_size=self.rollouts, num_envs=self.env.num_envs, device=device)
        self.agent = PhysHOIAgent(cfg.train, self.env.observation_space, self.env.action_space, self.memory,
                                  device, self.exp_dir, self.rofunc_logger)

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

    def inference(self):
        states, infos = self.env.reset()

        if self.cfg.task.env.playdataset:
            # # play dataset
            # while True:
            #     for t in range(self.env.max_episode_length):
            #         self.env.play_dataset_step(t)
            #         # reset env
            #         if self.env.terminated.any() or self.env.truncated.any():
            t = 0
            for _ in tqdm.trange(self.inference_steps):
                self.pre_interaction()
                with torch.no_grad():
                    # Obtain action from agent
                    action_state = self.env.get_dataset_step(t)
                    action_pos = action_state[:, 0]
                    action_pos = action_pos.repeat(self.env.num_envs, 1)
                    t += 1

                    # Interact with environment
                    next_states, rewards, terminated, truncated, infos = self.env.step(action_pos)

                    # # Reset the environment
                    if t >= self.env.max_episode_length:
                        states, infos = self.env.reset()

                    # if terminated.any() or truncated.any():
                    #     states, infos = self.env.reset()
                    # else:
                    #     states = next_states.clone()
        else:
            for _ in tqdm.trange(self.inference_steps):
                self.pre_interaction()
                with torch.no_grad():
                    # Obtain action from agent
                    actions, _ = self.agent.act(states, deterministic=True)  # TODO: check

                    # Interact with environment
                    next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                    # Reset the environment
                    if terminated.any() or truncated.any():
                        states, infos = self.env.reset()
                    else:
                        states = next_states.clone()
            # close the environment
            self.env.close()
            self.rofunc_logger.info('Inference complete.')
