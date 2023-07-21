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

import torch.nn as nn
from omegaconf import DictConfig

from rofunc.config.utils import omegaconf_to_dict
from rofunc.learning.RofuncRL.models.base_models import BaseMLP
from rofunc.learning.RofuncRL.models.utils import activation_func


class EmptyEncoder(nn.Module):
    def __init__(self):
        super(EmptyEncoder, self).__init__()

    def forward(self, x):
        return x


class BaseEncoder(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: int, output_dim: int, cfg_name: str = 'state_encoder'):
        super(BaseEncoder, self).__init__()

        self.cfg = cfg
        self.cfg_dict = omegaconf_to_dict(self.cfg)
        self.cfg_name = cfg_name

        self.input_dim = input_dim
        self.output_dim = output_dim

        if 'hidden_dims' in self.cfg_dict[cfg_name]:
            self.hidden_dims = self.cfg_dict[cfg_name]['hidden_dims']
        if 'activation' in self.cfg_dict[cfg_name]:
            self.activation = activation_func(self.cfg_dict[cfg_name]['activation'])

        if 'mlp_hidden_dims' in self.cfg_dict[cfg_name]:
            self.mlp_hidden_dims = self.cfg_dict[cfg_name]['mlp_hidden_dims']
        if 'mlp_activation' in self.cfg_dict[cfg_name]:
            self.mlp_activation = activation_func(self.cfg_dict[cfg_name]['mlp_activation'])


class MLPEncoder(BaseMLP):
    def __init__(self, cfg, input_dim, output_dim, cfg_name):
        super().__init__(cfg, input_dim, output_dim, cfg_name)
