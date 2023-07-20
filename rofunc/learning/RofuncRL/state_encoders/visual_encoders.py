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

        self.cnn_kernel_size = self.cfg_dict['encoder']['cnn_kernel_size']
        self.cnn_stride = self.cfg_dict['encoder']['cnn_stride']
        self.cnn_padding = self.cfg_dict['encoder']['cnn_padding']
        self.cnn_dilation = self.cfg_dict['encoder']['cnn_dilation']

        self.backbone_net = build_cnn(dims=[input_dim, *self.hidden_dims, output_dim], kernel_size=self.cnn_kernel_size,
                                      stride=self.cnn_stride, padding=self.cnn_padding, dilation=self.cnn_dilation,
                                      hidden_activation=self.activation)
        self.flatten = nn.Flatten(start_dim=1)
        self.output_net = nn.Linear(3 * 3 * self.hidden_dims[-1], output_dim)

        if self.cfg.use_init:
            init_layers(self.backbone_net, gain=1.0, init_type='kaiming_uniform')

    def forward(self, inputs):
        x = self.backbone_net(inputs)
        x = self.flatten(x)
        x = self.output_net(x)
        return x


class ResnetEncoder(BaseEncoder):
    def __init__(self, cfg, input_dim, output_dim):
        super().__init__(cfg, input_dim, output_dim)

        self.resnet_type = self.cfg_dict['encoder']['resnet_type']
        assert self.resnet_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        self.resnet_pre_trained = self.cfg_dict['encoder']['resnet_pre_trained']

        self.backbone_net = torch.hub.load('pytorch/vision', 'resnet18', weights="IMAGENET1K_V2",
                                           pretrained=self.resnet_pre_trained)

    def forward(self, inputs):
        x = self.net(inputs)
        return x


class ViTEncoder(BaseEncoder):
    def __init__(self, cfg, input_dim, output_dim):
        super().__init__(cfg, input_dim, output_dim)

        self.vit_type = self.cfg_dict['encoder']['vit_type']
        assert self.vit_type in ['']
        self.vit_pre_trained = self.cfg_dict['encoder']['vit_pre_trained']
