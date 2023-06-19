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


def init_layers(layers, gain=1.0, bias_const=1e-6):
    if not isinstance(layers, list):
        layers = [layers]
    for layer in layers:
        if isinstance(layer, nn.Linear):
            init_with_orthogonal(layer, gain, bias_const)


def init_with_orthogonal(layer, gain=1.0, bias_const=1e-6):
    """
    Parameter orthogonal initialization
    :param layer:
    :param gain:
    :param bias_const:
    :return:
    """
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.constant_(layer.bias, bias_const)
