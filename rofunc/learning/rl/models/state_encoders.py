import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn


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


class HomoGraphEncoder(BaseEncoder):
    def __init__(self, in_dim, hidden_dim, recurrent=False):
        super(HomoGraphEncoder, self).__init__(recurrent, hidden_dim, hidden_dim)
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


class ViTEncoder(BaseEncoder):
    def __init__(self):
        super().__init__()
