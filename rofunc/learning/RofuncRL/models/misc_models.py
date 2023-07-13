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
from torch import Tensor

from .base_models import BaseMLP
from .utils import init_layers


class ASEDiscEnc(BaseMLP):
    def __init__(self, cfg: DictConfig,
                 input_dim: int,
                 enc_output_dim: int,
                 disc_output_dim: int,
                 cfg_name: str):
        super().__init__(cfg, input_dim, disc_output_dim, cfg_name)
        self.encoder_layer = nn.Linear(self.mlp_hidden_dims[-1], enc_output_dim)
        self.disc_layer = self.output_net
        if self.cfg.use_init:
            init_layers(self.encoder_layer, gain=1.0)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)  # Same as self.get_disc(x)

    def get_enc(self, x: Tensor) -> Tensor:
        x = self.backbone_net(x)
        x = self.encoder_layer(x)
        return x

    def get_disc(self, x: Tensor) -> Tensor:
        x = self.backbone_net(x)
        x = self.disc_layer(x)
        return x
