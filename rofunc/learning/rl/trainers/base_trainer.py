import numpy as np
import torch
import copy
import datetime
import os
import time
from torch.utils.tensorboard import SummaryWriter
from typing import Union, Tuple, Dict, Optional

import tqdm
from rofunc.utils.logger.beauty_logger import BeautyLogger


class BaseTrainer:

    def __init__(self,
                 cfg,
                 env,
                 device: Optional[Union[str, torch.device]] = None):
        self.cfg = cfg
        self._env = env
        self._test_env = env
        # self._env.seed(self.cfg.Trainer.seed)
        # self._test_env.seed(2 ** 31 - 1 - self.cfg.Trainer.seed)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)

        '''Experiment log directory'''
        directory = self.cfg.get("Trainer", {}).get("log_directory", "")
        experiment_name = self.cfg.get("Trainer", {}).get("experiment_name", "")
        if not directory:
            directory = os.path.join(os.getcwd(), "runs")
        if not experiment_name:
            experiment_name = "{}_{}".format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"),
                                             self.__class__.__name__)
        self.experiment_dir = os.path.join(directory, experiment_name)

        '''Wandb and Tensorboard'''
        # setup Weights & Biases
        if self.cfg.get("experiment", {}).get("wandb", False):
            # save experiment config
            trainer_cfg = None  # TODO: check
            trainer_cfg = trainer_cfg if trainer_cfg is not None else {}
            try:
                models_cfg = {k: v.net._modules for (k, v) in self.models.items()}
            except AttributeError:
                models_cfg = {k: v._modules for (k, v) in self.models.items()}
            config = {**self.cfg, **trainer_cfg, **models_cfg}
            # set default values
            wandb_kwargs = copy.deepcopy(self.cfg.get("experiment", {}).get("wandb_kwargs", {}))
            wandb_kwargs.setdefault("name", os.path.split(self.experiment_dir)[-1])
            wandb_kwargs.setdefault("sync_tensorboard", True)
            wandb_kwargs.setdefault("config", {})
            wandb_kwargs["config"].update(config)
            # init Weights & Biases
            import wandb
            wandb.init(**wandb_kwargs)

        # main entry to log data for consumption and visualization by TensorBoard
        self.write_interval = self.cfg.get("experiment", {}).get("write_interval", 100)
        if self.write_interval > 0:
            self.writer = SummaryWriter(log_dir=self.experiment_dir)

        self.checkpoint_interval = self.cfg.get("experiment", {}).get("checkpoint_interval", 100)
        if self.checkpoint_interval > 0:
            os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)

        '''Rofunc logger'''
        # if self.cfg.get("experiment", {}).get("rofunc_logger", False):
        self.rofunc_logger = BeautyLogger(self.experiment_dir, 'rofunc.log', verbose=True)

        '''Misc variables'''
        self.step = 0
        self._episodes = 0
        self.start_time = None
        self.num_episodes = self.cfg.Trainer.num_episodes

    def train(self):
        for _ in tqdm.trange(self.num_episodes):
            self._episodes += 1
            self.train_episode()

        self.writer.close()
        self.rofunc_logger.info('Training complete.')

    def train_episode(self):
        pass

    def eval(self):
        pass

    def inference(self):
        pass
