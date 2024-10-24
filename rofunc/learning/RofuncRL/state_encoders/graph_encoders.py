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

from .base_encoders import BaseEncoder


class HomoGraphEncoder(BaseEncoder):
    def __init__(self, in_dim, hidden_dim):
        import dgl.nn.pytorch as dglnn

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
        import torch.nn.functional as F
        import dgl
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