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


class NnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


class BaseEncoder(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(BaseEncoder, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent


class CNNEncoder(BaseEncoder):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNEncoder, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        # input: [1, 16, 64, 64]
        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 8, 8, stride=4)), nn.ReLU(),  # [8, 15, 15]
            init_(nn.Conv2d(8, 16, 4, stride=2)), nn.ReLU(),  # [16, 6, 6]
            # init_(nn.Conv2d(16, 8, 3, stride=1)), nn.ReLU(),  # [8, 4, 4]
            Flatten(),  # [4608]
            init_(nn.Linear(16 * 6 * 6, hidden_size)), nn.ReLU())

        self.train()

    def forward(self, inputs):
        x = self.main(inputs)
        return x



