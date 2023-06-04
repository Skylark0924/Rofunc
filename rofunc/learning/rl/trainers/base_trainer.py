import copy
import datetime
import os
from typing import Union, Optional
import collections
from omegaconf import DictConfig
import numpy as np
import torch
import tqdm
import gym, gymnasium
from torch.utils.tensorboard import SummaryWriter

import rofunc as rf
from rofunc.utils.logger.beauty_logger import BeautyLogger
from rofunc.learning.rl.tasks.utils.env_wrappers import wrap_env


class BaseTrainer:

    def __init__(self,
                 cfg: DictConfig,
                 env: Union[gym.Env, gymnasium.Env],
                 device: Optional[Union[str, torch.device]] = None):
        self.cfg = cfg
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)

        '''Experiment log directory'''
        directory = self.cfg.get("Trainer", {}).get("experiment_directory", "")
        experiment_name = self.cfg.get("Trainer", {}).get("experiment_name", "")
        if not directory:
            directory = os.path.join(os.getcwd(), "runs")
        if not experiment_name:
            experiment_name = "{}_{}".format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"),
                                             self.__class__.__name__)
        self.experiment_dir = os.path.join(directory, experiment_name)

        # main entry to log data for consumption and visualization by TensorBoard
        self.write_interval = self.cfg.get("Trainer", {}).get("write_interval", 100)
        self.writer = SummaryWriter(log_dir=self.experiment_dir)

        '''Rofunc logger'''
        # if self.cfg.get("experiment", {}).get("rofunc_logger", False):
        self.rofunc_logger = BeautyLogger(self.experiment_dir, 'rofunc.log', verbose=True)

        '''Misc variables'''
        self.maximum_steps = self.cfg.Trainer.maximum_steps
        self.start_learning_steps = self.cfg.Trainer.start_learning_steps
        self.random_steps = self.cfg.Trainer.random_steps
        self.rollouts = self.cfg.Trainer.rollouts
        self._step = 0
        self._rollout = 0
        self._update_times = 0
        self.start_time = None

        self.env = wrap_env(env, logger=self.rofunc_logger)

    def train(self):
        """
        Main training loop.
        - Reset the environment
        - For each step:
            - Pre-interaction
            - Obtain action from agent
            - Interact with environment
            - Store transition
            - Reset the environment
            - Post-interaction
        - Close the environment
        """

        # reset env
        states, infos = self.env.reset()
        for _ in tqdm.trange(self.maximum_steps):
            self.pre_interaction()
            with torch.no_grad():
                # Obtain action from agent
                if self._step < self.random_steps:
                    actions = self.env.action_space.sample()  # sample random actions
                else:
                    actions, _ = self.agent.act(states)

                # Interact with environment
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # Store transition
                self.agent.store_transition(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                            terminated=terminated, truncated=truncated, infos=infos)

                # Reset the environment
                if terminated.any() or truncated.any():
                    states, infos = self.env.reset()
                else:
                    states = next_states.clone()

                self._step += 1
            self.post_interaction()
        # close the environment
        self.env.close()
        # close the logger
        self.writer.close()
        self.rofunc_logger.info('Training complete.')

    def eval(self):
        # reset env
        states, infos = self.env.reset()

        for _ in tqdm.trange(self.num_episodes):
            self._episodes += 1

            with torch.no_grad():
                # Obtain action from agent
                actions = self.agent.act(states)

                # Interact with environment
                next_states, rewards, dones, infos = self.env.step(actions)

                # Store transition
                self.agent.store_transition(states, actions, rewards, next_states, dones)

                # Reset the environment
                if dones.any():
                    states, infos = self.env.reset()
                else:
                    states = next_states.copy()

        # close the environment
        self.env.close()

        # close the logger
        self.writer.close()
        self.rofunc_logger.info('Training complete.')

    def inference(self):
        pass

    def pre_interaction(self):
        pass

    def post_interaction(self):
        self._rollout += 1

        # Update agent
        if not self._rollout % self.rollouts and self._step >= self.start_learning_steps:
            self.agent.update_net()
            self._update_times += 1
            self.rofunc_logger.info(f'Update {self._update_times} times.')

        # Update best models and tensorboard
        if not self._step % self.write_interval and self.write_interval > 0 and self._step > 1:
            # update best models
            reward = np.mean(self.agent.tracking_data.get("Reward / Total reward (mean)", -2 ** 31))
            if reward > self.agent.checkpoint_best_modules["reward"]:
                self.agent.checkpoint_best_modules["timestep"] = self._step
                self.agent.checkpoint_best_modules["reward"] = reward
                self.agent.checkpoint_best_modules["saved"] = False
                self.agent.checkpoint_best_modules["modules"] = {k: copy.deepcopy(self.agent._get_internal_value(v)) for k, v
                                                                 in self.agent.checkpoint_modules.items()}

            # Update tensorboard
            self.write_tensorboard()

        # Save checkpoints
        if not self._step % self.agent.checkpoint_interval and self.agent.checkpoint_interval > 0 and self._step > 1:
            self.agent.save_ckpt(os.path.join(self.agent.checkpoint_dir, f"ckpt_{self._step}.pth"))

    def write_tensorboard(self):
        for k, v in self.agent.tracking_data.items():
            if k.endswith("(min)"):
                self.writer.add_scalar(k, np.min(v), self._step)
            elif k.endswith("(max)"):
                self.writer.add_scalar(k, np.max(v), self._step)
            else:
                self.writer.add_scalar(k, np.mean(v), self._step)
        # reset data containers for next iteration
        self.agent.track_rewards.clear()
        self.agent.track_timesteps.clear()
        self.agent.tracking_data.clear()
