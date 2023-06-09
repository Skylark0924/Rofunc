import copy
import datetime
import os
from typing import Union, Optional

import gym
import gymnasium
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

import rofunc as rf
from rofunc.learning.rl.processors.normalizers import Normalization
from rofunc.learning.rl.tasks.utils.env_wrappers import wrap_env
from rofunc.utils.file.internet import reserve_sock_addr
from rofunc.utils.logger.beauty_logger import BeautyLogger


class BaseTrainer:

    def __init__(self,
                 cfg: DictConfig,
                 env: Union[gym.Env, gymnasium.Env],
                 device: Optional[Union[str, torch.device]] = None):
        self.cfg = cfg
        self.agent = None
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)

        '''Experiment log directory'''
        directory = self.cfg.Trainer.experiment_directory
        experiment_name = self.cfg.Trainer.experiment_name
        if not directory:
            directory = os.path.join(os.getcwd(), "runs")
        if not experiment_name:
            experiment_name = "RofuncRL_{}_{}_{}".format(self.__class__.__name__, self.cfg.Trainer.task_name,
                                                         datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"))
        self.experiment_dir = os.path.join(directory, experiment_name)
        rf.utils.create_dir(self.experiment_dir)

        '''Rofunc logger'''
        self.rofunc_logger = BeautyLogger(self.experiment_dir, 'rofunc.log',
                                          verbose=self.cfg.Trainer.rofunc_logger_kwargs.verbose)
        self.rofunc_logger.info(f"Configurations:\n{OmegaConf.to_yaml(self.cfg)}")

        '''TensorBoard'''
        # main entry to log data for consumption and visualization by TensorBoard
        self.write_interval = self.cfg.Trainer.write_interval
        self.writer = SummaryWriter(log_dir=self.experiment_dir)
        tb = program.TensorBoard()
        # Find a free port
        with reserve_sock_addr() as (h, p):
            argv = ['tensorboard', f"--logdir={self.experiment_dir}", f"--port={p}"]
            tb_extra_args = os.getenv('TB_EXTRA_ARGS', "")
            if tb_extra_args:
                argv += tb_extra_args.split(' ')
            tb.configure(argv)
        # Launch TensorBoard
        url = tb.launch()
        self.rofunc_logger.info(f"Tensorflow listening on {url}")

        '''Misc variables'''
        self.maximum_steps = self.cfg.Trainer.maximum_steps
        self.start_learning_steps = self.cfg.Trainer.start_learning_steps
        self.random_steps = self.cfg.Trainer.random_steps
        self.rollouts = self.cfg.Trainer.rollouts
        self._step = 0
        self._rollout = 0
        self._update_times = 0
        self.start_time = None
        self.inference_steps = self.cfg.Trainer.inference_steps

        '''Environment'''
        self.env = wrap_env(env, logger=self.rofunc_logger)

        '''Normalization'''
        self.state_norm = Normalization(shape=self.env.observation_space.shape[0], device=device)

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

            with torch.no_grad():
                # Store transition
                self.agent.store_transition(states=states, actions=actions, next_states=next_states, rewards=rewards,
                                            terminated=terminated, truncated=truncated, infos=infos)
            self._step += 1
            self.post_interaction()

            with torch.no_grad():
                # Reset the environment
                if terminated.any() or truncated.any():
                    states, infos = self.env.reset()
                else:
                    states = next_states.clone()

        # close the environment
        self.env.close()
        # close the logger
        self.writer.close()
        self.rofunc_logger.info('Training complete.')

    def eval(self):
        # TODO: implement evaluation
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
        # reset env
        states, infos = self.env.reset()
        for _ in tqdm.trange(self.inference_steps):
            with torch.no_grad():
                # states = self.state_norm(states)
                # Obtain action from agent
                actions, _ = self.agent.act(states, deterministic=True)  # TODO: check

                # Interact with environment
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                # next_states = self.state_norm(next_states)

                # Reset the environment
                if terminated.any() or truncated.any():
                    states, infos = self.env.reset()
                else:
                    states = next_states.clone()
        # close the environment
        self.env.close()
        self.rofunc_logger.info('Inference complete.')

    def pre_interaction(self):
        pass

    def post_interaction(self):
        self._rollout += 1

        # Update agent
        if not self._rollout % self.rollouts and self._step >= self.start_learning_steps:
            for model in self.agent.models.values():
                model.train(True)
            self.agent.update_net()
            for model in self.agent.models.values():
                model.train(False)
            self._update_times += 1
            self.rofunc_logger.info(f'Update {self._update_times} times.', local_verbose=False)

        # Update best models and tensorboard
        if not self._step % self.write_interval and self.write_interval > 0 and self._step > 1:
            # update best models
            reward = np.mean(self.agent.tracking_data.get("Reward / Total reward (mean)", -2 ** 31))
            if reward > self.agent.checkpoint_best_modules["reward"]:
                self.agent.checkpoint_best_modules["timestep"] = self._step
                self.agent.checkpoint_best_modules["reward"] = reward
                self.agent.checkpoint_best_modules["saved"] = False
                self.agent.checkpoint_best_modules["modules"] = {k: copy.deepcopy(self.agent._get_internal_value(v)) for
                                                                 k, v in self.agent.checkpoint_modules.items()}

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
