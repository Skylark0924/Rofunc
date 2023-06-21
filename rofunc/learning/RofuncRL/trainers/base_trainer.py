"""
 Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import copy
import datetime
import os
import random
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
from rofunc.learning.RofuncRL.processors.normalizers import Normalization
from rofunc.learning.RofuncRL.tasks.utils.env_wrappers import wrap_env
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
        exp_name = self.cfg.Trainer.experiment_name
        directory = os.path.join(os.getcwd(), "runs") if not directory else directory
        exp_name = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f") if not exp_name else exp_name
        # env_name = env.cfg['name'] if hasattr(env, 'cfg') else env.envs[0].spec.id if hasattr(env, 'envs') else env.spec.id
        env_name = 'ASE'
        exp_name = "RofuncRL_{}_{}_{}".format(self.__class__.__name__, env_name, exp_name)
        self.exp_dir = os.path.join(directory, exp_name)
        rf.utils.create_dir(self.exp_dir)

        '''Rofunc logger'''
        self.rofunc_logger = BeautyLogger(self.exp_dir, verbose=self.cfg.Trainer.rofunc_logger_kwargs.verbose)
        self.rofunc_logger.info(f"Configurations:\n{OmegaConf.to_yaml(self.cfg)}")

        '''TensorBoard'''
        # main entry to log data for consumption and visualization by TensorBoard
        self.write_interval = self.cfg.Trainer.write_interval
        self.writer = SummaryWriter(log_dir=self.exp_dir)
        tb = program.TensorBoard()
        # Find a free port
        with reserve_sock_addr() as (h, p):
            argv = ['tensorboard', f"--logdir={self.exp_dir}", f"--port={p}"]
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
        self.eval_steps = self.cfg.Trainer.eval_steps
        self.inference_steps = self.cfg.Trainer.inference_steps

        '''Environment'''
        env.device = self.device
        self.env = wrap_env(env, logger=self.rofunc_logger, seed=self.cfg.Trainer.seed)
        self.eval_env = wrap_env(env, logger=self.rofunc_logger, seed=random.randint(0, 10000))
        self.rofunc_logger.info(f"Environment:\n  "
                                f"  action_space: {self.env.action_space.shape}\n  "
                                f"  observation_space: {self.env.observation_space.shape}\n  "
                                f"  num_envs: {self.env.num_envs}")

        '''Normalization'''
        self.state_norm = Normalization(shape=self.env.observation_space.shape[0], device=device)

    def setup_wandb(self):
        # setup Weights & Biases
        if self.cfg.get("Trainer", {}).get("wandb", False):
            # save experiment config
            trainer_cfg = None  # TODO: check
            trainer_cfg = trainer_cfg if trainer_cfg is not None else {}
            try:
                models_cfg = {k: v.net._modules for (k, v) in self.agent.models.items()}
            except AttributeError:
                models_cfg = {k: v._modules for (k, v) in self.agent.models.items()}
            config = {**self.cfg, **trainer_cfg, **models_cfg}
            # set default values
            wandb_kwargs = copy.deepcopy(self.cfg.get("Trainer", {}).get("wandb_kwargs", {}))
            wandb_kwargs.setdefault("name", os.path.split(self.exp_dir)[-1])
            wandb_kwargs.setdefault("sync_tensorboard", True)
            wandb_kwargs.setdefault("config", {})
            wandb_kwargs["config"].update(config)
            # init Weights & Biases
            import wandb
            wandb.init(**wandb_kwargs)

    def get_action(self, states):
        if self._step < self.random_steps:
            actions = torch.tensor([self.env.action_space.sample() for _ in range(self.env.num_envs)]).to(self.device)
        else:
            actions, _ = self.agent.act(states)
        return actions

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
        with tqdm.trange(self.maximum_steps, ncols=80, colour='green') as self.t_bar:
            for _ in self.t_bar:
                self.pre_interaction()
                # Obtain action from agent
                with torch.no_grad():
                    actions = self.get_action(states)

                # Interact with environment
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                with torch.no_grad():
                    # Store transition
                    self.agent.store_transition(states=states, actions=actions, next_states=next_states,
                                                rewards=rewards,
                                                terminated=terminated, truncated=truncated, infos=infos)
                self.post_interaction()
                self._step += 1

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

    def pre_interaction(self):
        pass

    def post_interaction(self):
        # Update best models and tensorboard
        if not self._step % self.write_interval and self.write_interval > 0:
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

            # Update tqdm bar message
            self.t_bar.set_postfix_str(f"Rew/Best: {reward:.2f}/{self.agent.checkpoint_best_modules['reward']:.2f}")

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

    def eval(self):
        # reset env
        states, infos = self.eval_env.reset()
        for _ in tqdm.trange(self.eval_steps):
            with torch.no_grad():
                # Obtain action from agent
                actions, _ = self.agent.act(states, deterministic=True)  # TODO: check

                # Interact with environment
                next_states, rewards, terminated, truncated, infos = self.eval_env.step(actions)

                # Reset the environment
                if terminated.any() or truncated.any():
                    states, infos = self.eval_env.reset()
                else:
                    states = next_states.clone()
        # close the environment
        self.eval_env.close()
        self.rofunc_logger.info('Evaluation complete.')

    def inference(self):
        # reset env
        states, infos = self.env.reset()
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
