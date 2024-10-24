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
