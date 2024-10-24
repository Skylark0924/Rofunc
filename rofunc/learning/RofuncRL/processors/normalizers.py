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
from rofunc.learning.RofuncRL.models.utils import get_space_dim


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, device):  # shape:the dimension of input data
        self.shape = get_space_dim(shape)
        self.n = 0
        self.mean = torch.zeros(self.shape).to(device)
        self.S = torch.zeros(self.shape).to(device)
        self.std = torch.sqrt(self.S).to(device)

    def train(self, x):
        if isinstance(self.shape, int):
            if len(x.shape) == 2:
                batch_size = x.shape[0]
                x = torch.sum(x, dim=0) / batch_size
        else:
            if len(x.shape) == len(self.shape) + 1:
                batch_size = x.shape[0]
                x = torch.sum(x, dim=0) / batch_size

        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.clone()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape, device):
        """
        State Normalization
        :param shape:
        """
        self.device = device
        self.running_ms = RunningMeanStd(shape=shape, device=self.device)

    def __call__(self, x, train=False):
        # Whether to update the mean and std, during the evaluating, update=False
        if train:
            self.running_ms.train(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma, device):
        """
        Reward Normalization & Reward Scaling
        :param shape:
        :param gamma:
        :param device:
        """
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.device = device
        self.running_ms = RunningMeanStd(shape=self.shape, device=self.device)
        self.R = torch.zeros(self.shape).to(self.device)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = torch.zeros(self.shape).to(self.device)
