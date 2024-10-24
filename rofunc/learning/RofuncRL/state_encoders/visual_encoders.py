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
import torch
import torch.nn as nn

from rofunc.learning.RofuncRL.models.utils import activation_func
from rofunc.learning.RofuncRL.models.utils import build_mlp, build_cnn, init_layers
from rofunc.learning.RofuncRL.state_encoders.base_encoders import BaseEncoder


class CNNEncoder(BaseEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.cnn_kernel_size = self.cfg_dict[self.cfg_name]['cnn_args']['cnn_kernel_size']
        self.cnn_stride = self.cfg_dict[self.cfg_name]['cnn_args']['cnn_stride']
        self.cnn_padding = self.cfg_dict[self.cfg_name]['cnn_args']['cnn_padding']
        self.cnn_dilation = self.cfg_dict[self.cfg_name]['cnn_args']['cnn_dilation']
        self.cnn_hidden_dims = self.cfg_dict[self.cfg_name]['cnn_args']['cnn_hidden_dims']
        self.cnn_activation = activation_func(self.cfg_dict[self.cfg_name]['cnn_args']['cnn_activation'])
        self.cnn_pooling = self.cfg_dict[self.cfg_name]['cnn_args']['cnn_pooling']
        self.cnn_pooling_args = self.cfg_dict[self.cfg_name]['cnn_args']['cnn_pooling_args']
        self.mlp_inp_dims = self.cfg_dict[self.cfg_name]['cnn_args']['mlp_inp_dims']
        self.mlp_hidden_dims = self.cfg_dict[self.cfg_name]['cnn_args']['mlp_hidden_dims']
        self.mlp_activation = activation_func(self.cfg_dict[self.cfg_name]['cnn_args']['mlp_activation'])

        self.backbone_net = build_cnn(dims=[self.input_dim, *self.cnn_hidden_dims], kernel_size=self.cnn_kernel_size,
                                      stride=self.cnn_stride, padding=self.cnn_padding, dilation=self.cnn_dilation,
                                      hidden_activation=self.cnn_activation, pooling=self.cnn_pooling,
                                      pooling_args=self.cnn_pooling_args)

        self.flatten = nn.Flatten(start_dim=1)

        self.output_net = build_mlp(dims=[self.mlp_inp_dims, *self.mlp_hidden_dims, self.output_dim],
                                    hidden_activation=self.mlp_activation)

        self.checkpoint_modules = {'backbone_net': self.backbone_net, 'output_net': self.output_net}

        if self.cfg.use_init:
            init_layers(self.backbone_net, gain=1.0, init_type='kaiming_uniform')
            init_layers(self.output_net, gain=1.0)

        self.freeze_net_list = [self.backbone_net, self.output_net]
        self.set_up()

    def forward(self, inputs):
        x = self.backbone_net(inputs)
        x = self.flatten(x)
        x = self.output_net(x)
        return x


class ResnetEncoder(BaseEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.sub_type = self.cfg_dict[self.cfg_name]['resnet_args']['sub_type']
        assert self.sub_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        self.use_pretrained = self.cfg_dict[self.cfg_name]['use_pretrained']

        self.backbone_net = torch.hub.load('pytorch/vision', self.sub_type, num_classes=self.output_dim,
                                           pretrained=self.use_pretrained)

        self.freeze_net_list = [self.backbone_net]
        self.set_up()

    def forward(self, inputs):
        x = self.backbone_net(inputs)
        return x


class ViTEncoder(BaseEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.sub_type = self.cfg_dict[self.cfg_name]['vit_args']['sub_type']
        assert self.sub_type in ['vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32']
        self.use_pretrained = self.cfg_dict[self.cfg_name]['use_pretrained']

        self.backbone_net = torch.hub.load('pytorch/vision', self.sub_type, num_classes=self.output_dim,
                                           pretrained=self.use_pretrained)

        self.freeze_net_list = [self.backbone_net]
        self.set_up()

    def forward(self, inputs):
        x = self.backbone_net(inputs)
        return x


if __name__ == '__main__':
    from omegaconf import DictConfig
    import rofunc as rf

    cfg = DictConfig({'use_init': True, 'state_encoder': {'inp_channels': 4, 'out_channels': 512,
                                                          'use_pretrained': True,
                                                          'freeze': False,
                                                          'model_ckpt': 'test.ckpt',
                                                          'cnn_args': {
                                                              'cnn_structure': ['conv', 'relu', 'conv', 'relu', 'pool'],
                                                              'cnn_kernel_size': [8, 4],
                                                              'cnn_stride': 1,
                                                              'cnn_padding': 1,
                                                              'cnn_dilation': 1,
                                                              'cnn_hidden_dims': [32, 64],
                                                              'cnn_activation': 'relu',
                                                              'cnn_pooling': None,  # ['max', 'avg']
                                                              'cnn_pooling_args': {
                                                                  'cnn_pooling_kernel_size': 2,
                                                                  'cnn_pooling_stride': 2,
                                                                  'cnn_pooling_padding': 0,
                                                                  'cnn_pooling_dilation': 1},
                                                              'mlp_inp_dims': 215296,
                                                              'mlp_hidden_dims': [512],
                                                              'mlp_activation': 'relu',
                                                          }}})
    model = CNNEncoder(cfg=cfg).to('cuda:0')
    print(model)

    # cfg = DictConfig(
    #     {'use_init': True,
    #      'state_encoder': {'inp_channels': 4, 'out_channels': 512, 'resnet_args': {'sub_type': 'resnet18'},
    #                        'use_pretrained': False}})
    # model = ResnetEncoder(cfg=cfg)
    # print(model)

    # cfg = DictConfig({'use_init': True,
    #                   'state_encoder': {'inp_channels': 4, 'out_channels': 512, 'vit_args': {'sub_type': 'vit_b_16'},
    #                                     'use_pretrained': False}})
    # model = ViTEncoder(cfg=cfg)
    # print(model)

    inp_latent_vector = torch.randn(32, 4, 64, 64).to('cuda:0')  # [B, C, H, W]
    gt_latent_vector = torch.randn(32, 512).to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(10000):
        out_latent_vector = model(inp_latent_vector)
        # predicted_latent_vector = out_latent_vector.last_hidden_state[:, -1:, :]
        # print(out_latent_vector)

        loss = nn.MSELoss()(out_latent_vector, gt_latent_vector)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            model.save_ckpt('test.ckpt')
            rf.logger.beauty_print('Save ckpt')

        print(loss)
