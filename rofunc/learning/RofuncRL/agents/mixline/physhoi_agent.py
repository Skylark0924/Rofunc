# Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import Callable, Union, Tuple, Optional

import rofunc as rf
from rofunc.learning.RofuncRL.agents.base_agent import BaseAgent
from rofunc.learning.RofuncRL.agents.mixline.amp_agent import AMPAgent
from rofunc.learning.RofuncRL.models.base_models import BaseMLP
from rofunc.learning.RofuncRL.utils.memory import Memory


class PhysHOIAgent(AMPAgent):
    """
    PhysHOI agent
    """

    def __init__(
        self,
        cfg: DictConfig,
        observation_space: Optional[
            Union[int, Tuple[int], gym.Space, gymnasium.Space, List]
        ],
        action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        experiment_dir: Optional[str] = None,
        rofunc_logger: Optional[rf.logger.BeautyLogger] = None,
        amp_observation_space: Optional[
            Union[int, Tuple[int], gym.Space, gymnasium.Space]
        ] = None,
        motion_dataset: Optional[Union[Memory, Tuple[Memory]]] = None,
        replay_buffer: Optional[Union[Memory, Tuple[Memory]]] = None,
        collect_reference_motions: Optional[Callable[[int], torch.Tensor]] = None,
    ):
        """
        :param cfg: Configuration
        :param observation_space: Observation space
        :param action_space: Action space
        :param memory: Memory for storing transitions
        :param device: Device on which the torch tensor is allocated
        :param experiment_dir: Directory where experiment outputs are saved
        :param rofunc_logger: Rofunc logger
        :param amp_observation_space: cfg["env"]["numAMPObsSteps"] * NUM_AMP_OBS_PER_STEP
        :param motion_dataset: Motion dataset
        :param replay_buffer: Replay buffer
        :param collect_reference_motions: Function for collecting reference motions
        """
        super().__init__(
            cfg,
            observation_space,
            action_space,
            memory,
            device,
            experiment_dir,
            rofunc_logger,
            amp_observation_space,
            motion_dataset,
            replay_buffer,
            collect_reference_motions,
        )

    def collect_reference_motions(self, num_motions: int) -> torch.Tensor:
        """
        Collect reference motions
        :param num_motions: Number of motions to collect
        :return: Reference motions
        """
        return self.motion_dataset.sample(num_motions)
