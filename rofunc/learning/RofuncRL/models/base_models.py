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

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from rofunc.config.utils import omegaconf_to_dict
from rofunc.learning.RofuncRL.models.utils import build_mlp, init_layers, activation_func


class BaseMLP(nn.Module):
    def __init__(self, cfg: DictConfig,
                 input_dim: int,
                 output_dim: int,
                 cfg_name: str):
        super().__init__()

        self.cfg = cfg
        self.cfg_dict = omegaconf_to_dict(self.cfg)
        self.mlp_hidden_dims = self.cfg_dict[cfg_name]['mlp_hidden_dims']
        self.mlp_activation = activation_func(self.cfg_dict[cfg_name]['mlp_activation'])

        self.backbone_net = build_mlp(dims=[input_dim, *self.mlp_hidden_dims], hidden_activation=self.mlp_activation)
        self.output_net = nn.Linear(self.mlp_hidden_dims[-1], output_dim)
        if self.cfg.use_init:
            init_layers(self.backbone_net, gain=1.0)
            init_layers(self.output_net, gain=1.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone_net(x)
        x = self.output_net(x)
        return x
