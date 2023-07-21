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
import torch
import torch.nn as nn

from rofunc.learning.RofuncRL.models.utils import build_mlp, build_cnn, init_layers
from rofunc.learning.RofuncRL.state_encoders.base_encoders import BaseEncoder


class CNNEncoder(BaseEncoder):
    def __init__(self, cfg, input_dim, output_dim):
        super().__init__(cfg, input_dim, output_dim)

        self.cnn_kernel_size = self.cfg_dict[self.cfg_name]['cnn_kernel_size']
        self.cnn_stride = self.cfg_dict[self.cfg_name]['cnn_stride']
        self.cnn_padding = self.cfg_dict[self.cfg_name]['cnn_padding']
        self.cnn_dilation = self.cfg_dict[self.cfg_name]['cnn_dilation']

        self.backbone_net = build_cnn(dims=[input_dim, *self.hidden_dims], kernel_size=self.cnn_kernel_size,
                                      stride=self.cnn_stride, padding=self.cnn_padding, dilation=self.cnn_dilation,
                                      hidden_activation=self.activation)

        self.flatten = nn.Flatten(start_dim=1)

        final_kernel_size = self.cnn_kernel_size[-1] if isinstance(self.cnn_kernel_size, list) else self.cnn_kernel_size
        self.output_net = build_mlp(
            dims=[final_kernel_size * final_kernel_size * self.hidden_dims[-1], *self.mlp_hidden_dims, output_dim],
            hidden_activation=self.mlp_activation)

        if self.cfg.use_init:
            init_layers(self.backbone_net, gain=1.0, init_type='kaiming_uniform')
            init_layers(self.output_net, gain=1.0)

    def forward(self, inputs):
        x = self.backbone_net(inputs)
        x = self.flatten(x)
        x = self.output_net(x)
        return x


class ResnetEncoder(BaseEncoder):
    def __init__(self, cfg, input_dim, output_dim):
        super().__init__(cfg, input_dim, output_dim)

        self.resnet_type = self.cfg_dict[self.cfg_name]['resnet_type']
        assert self.resnet_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        self.use_pretrained = self.cfg_dict[self.cfg_name]['use_pretrained']

        self.backbone_net = torch.hub.load('pytorch/vision', self.resnet_type, num_classes=output_dim,
                                           pretrained=self.use_pretrained)

    def forward(self, inputs):
        x = self.backbone_net(inputs)
        return x


class ViTEncoder(BaseEncoder):
    def __init__(self, cfg, input_dim, output_dim):
        super().__init__(cfg, input_dim, output_dim)

        self.vit_type = self.cfg_dict[self.cfg_name]['vit_type']
        assert self.vit_type in ['vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32']
        self.use_pretrained = self.cfg_dict[self.cfg_name]['use_pretrained']

        self.backbone_net = torch.hub.load('pytorch/vision', self.vit_type, num_classes=output_dim,
                                           pretrained=self.use_pretrained)

    def forward(self, inputs):
        x = self.backbone_net(inputs)
        return x


if __name__ == '__main__':
    from omegaconf import DictConfig

    # cfg = DictConfig(
    #     {'use_init': True, 'state_encoder': {'cnn_kernel_size': 3, 'cnn_stride': 1, 'cnn_padding': 1, 'cnn_dilation': 1,
    #                                    'hidden_dims': [32, 64, 128], 'activation': 'relu', 'mlp_hidden_dims': [128],
    #                                    'mlp_activation': 'relu'}})
    # model = CNNEncoder(cfg=cfg, input_dim=3, output_dim=128)
    # print(model)

    # cfg = DictConfig({'use_init': True, 'state_encoder': {'resnet_type': 'resnet18', 'use_pretrained': False}})
    # model = ResnetEncoder(cfg=cfg, input_dim=3, output_dim=128)
    # print(model)

    cfg = DictConfig({'use_init': True, 'state_encoder': {'vit_type': 'vit_b_16', 'use_pretrained': False}})
    model = ViTEncoder(cfg=cfg, input_dim=3, output_dim=128)
    print(model)
