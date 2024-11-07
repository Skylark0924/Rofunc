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

from typing import Union, List
from collections.abc import Mapping

import gym
import gymnasium
import torch.nn as nn


def build_mlp(dims: [int], hidden_activation: nn = nn.ReLU, output_activation: nn = None) -> nn.Sequential:
    """
    Build multi-Layer perceptron
    :param dims: layer dimensions, including input and output layers
    :param hidden_activation: activation function for hidden layers
    :param output_activation: activation function for output layer
    :return:
    """
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(hidden_activation)
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def build_cnn(dims: [int], kernel_size: Union[int, tuple, List], stride: Union[int, tuple, List] = 1,
              padding: Union[int, tuple, List] = 0, dilation: Union[int, tuple, List] = 1,
              hidden_activation: nn = nn.ReLU, output_activation: nn = None, pooling=None,
              pooling_args: dict = None) -> nn.Sequential:
    """
    Build convolutional neural network
    :param dims: layer dimensions, including input and output layers
    :param kernel_size: kernel size for convolutional layers
    :param stride: stride controls the stride for the cross-correlation
    :param padding: padding controls the amount of padding applied to the input
    :param dilation: dilation controls the spacing between the kernel points
    :param hidden_activation: activation function for hidden layers
    :param output_activation: activation function for output layer
    :param pooling: pooling layer type, ['max', 'avg', 'none']
    :param pooling_args: pooling layer arguments

    The parameters kernel_size, stride, padding, dilation can either be:
    - a single int: in which case the same value is used for the height and width dimension
    - a tuple of two ints: in which case, the first int is used for the height dimension, and the second int for the width dimension
    :return:
    """
    if isinstance(kernel_size, int) or isinstance(kernel_size, tuple):
        kernel_size = [kernel_size] * (len(dims) - 1)
    if isinstance(stride, int) or isinstance(stride, tuple):
        stride = [stride] * (len(dims) - 1)
    if isinstance(padding, int) or isinstance(padding, tuple):
        padding = [padding] * (len(dims) - 1)
    if isinstance(dilation, int) or isinstance(dilation, tuple):
        dilation = [dilation] * (len(dims) - 1)

    if pooling is not None:
        if isinstance(pooling_args['cnn_pooling_kernel_size'], int) or isinstance(kernel_size, tuple):
            pooling_kernel_size = [pooling_args['cnn_pooling_kernel_size']] * (len(dims) - 1)
        if isinstance(pooling_args['cnn_pooling_stride'], int) or isinstance(stride, tuple):
            pooling_stride = [pooling_args['cnn_pooling_stride']] * (len(dims) - 1)
        if isinstance(pooling_args['cnn_pooling_padding'], int) or isinstance(padding, tuple):
            pooling_padding = [pooling_args['cnn_pooling_padding']] * (len(dims) - 1)
        if isinstance(pooling_args['cnn_pooling_dilation'], int) or isinstance(dilation, tuple):
            pooling_dilation = [pooling_args['cnn_pooling_dilation']] * (len(dims) - 1)

        if pooling == 'max':
            pooling_method = nn.MaxPool2d
        elif pooling == 'avg':
            pooling_method = nn.AvgPool2d

    layers = []
    for i in range(len(dims) - 1):
        layers.append(
            nn.Conv2d(dims[i], dims[i + 1], kernel_size=kernel_size[i], stride=stride[i], padding=padding[i],
                      dilation=dilation[i]))
        layers.append(hidden_activation)
        if pooling is not None:
            layers.append(
                pooling_method(kernel_size=pooling_kernel_size[i], stride=pooling_stride[i], padding=pooling_padding[i],
                               dilation=pooling_dilation[i]))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def activation_func(activation: str) -> nn:
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'elu':
        return nn.ELU()
    else:
        raise NotImplementedError


def init_layers(layers, gain=1.0, bias_const=1e-6, init_type='orthogonal'):
    # For more information about the initialization, please refer to https://pytorch.org/docs/stable/nn.init.html
    init_type_map = {
        "orthogonal": nn.init.orthogonal_,
        "xavier_normal": nn.init.xavier_normal_,
        "xavier_uniform": nn.init.xavier_uniform_,
        "kaiming_normal": nn.init.kaiming_normal_,
        "kaiming_uniform": nn.init.kaiming_uniform_,
        "normal": nn.init.normal_,
    }
    torch_init = init_type_map[init_type]

    for layer in layers.children():
        if isinstance(layer, nn.Linear):
            torch_init(layer.weight, gain=gain)
        elif isinstance(layer, nn.Conv2d):
            assert init_type in ['kaiming_normal', 'kaiming_uniform', 'xavier_uniform']
            torch_init(layer.weight)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.weight, 1)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight, 1)

        if hasattr(layer, 'bias'):
            nn.init.constant_(layer.bias, bias_const)


def get_space_dim(space):
    if isinstance(space, int):
        dim = space
    elif isinstance(space, tuple) or isinstance(space, list):
        dim = 0
        for i in range(len(space)):
            dim += get_space_dim(space[i])
    elif isinstance(space, Mapping):
        dim = get_space_dim(space["policy"])
    elif isinstance(space, gym.Space) or isinstance(space, gymnasium.Space):
        dim = space.shape
        if isinstance(dim, tuple) and len(dim) == 1:
            dim = dim[0]
    else:
        raise NotImplementedError
    return dim
