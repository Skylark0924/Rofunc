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

import dgl.nn.pytorch as dglnn
import torch.nn as nn

from .base_encoders import BaseEncoder


class HomoGraphEncoder(BaseEncoder):
    def __init__(self, in_dim, hidden_dim):
        super(HomoGraphEncoder, self).__init__(hidden_dim)
        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                        constant_(x, 0), nn.init.calculate_gain('relu'))

        # self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, activation=F.relu)
        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, activation=F.relu)
        # self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim, activation=F.relu)

        num_heads = 3
        self.conv1 = dglnn.GATConv(in_dim, hidden_dim, num_heads=num_heads)
        self.conv2 = dglnn.GATConv(hidden_dim * num_heads, hidden_dim, 1)
        # self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim, activation=F.relu)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, g, inputs):
        # 应用图卷积和激活函数
        h = self.conv1(g, inputs)
        h = h.view(-1, h.size(1) * h.size(2))
        h = F.elu(h)
        h = self.conv2(g, h)

        h = h.squeeze()
        # h = self.conv3(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # 使用平均读出计算图表示
            hg = dgl.mean_nodes(g, 'h')
            hg = self.linear(hg)

            return hg


class HeteroGraphEncoder(BaseEncoder):
    def __init__(self):
        super(HeteroGraphEncoder, self).__init__()