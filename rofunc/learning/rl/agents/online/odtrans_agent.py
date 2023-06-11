from typing import Union, Tuple, Optional

import gym
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from rofunc.learning.rl.processors.standard_scaler import RunningStandardScaler
from rofunc.learning.rl.processors.schedulers import KLAdaptiveRL

import rofunc as rf
from rofunc.learning.rl.agents.base_agent import BaseAgent
from rofunc.learning.rl.models.actor_models import ActorPPO_Beta, ActorPPO_Gaussian
from rofunc.learning.rl.models.critic_models import Critic
from rofunc.learning.rl.processors.normalizers import empty_preprocessor
from rofunc.learning.rl.utils.memory import Memory


class ODTransAgent(BaseAgent):
    def __init__(self,
                 cfg: DictConfig,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 experiment_dir: Optional[str] = None,
                 rofunc_logger: Optional[rf.utils.BeautyLogger] = None):
        """
        Online Decision Transformer (ODTrans) Agent
        "Online Decision Transformer". Qinqing Zheng. et al. https://arxiv.org/abs/2202.05607
        Rofunc documentation: https://rofunc.readthedocs.io/en/latest/lfd/RofuncRL/ODTrans.html
        :param cfg:
        :param observation_space:
        :param action_space:
        :param memory:
        :param device:
        :param experiment_dir:
        :param rofunc_logger:
        """
        super().__init__(cfg, observation_space, action_space, memory, device, experiment_dir, rofunc_logger)
