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

# import torch
# import torch.nn as nn
#
# from .utils import build_mlp, init_with_orthogonal
#
#
# class BaseQNet(nn.Module):
#     def __init__(self, state_dim: int, action_dim: int):
#         super().__init__(state_dim, action_dim)
#         self.explore_rate = 0.125
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])
#
#         self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
#         self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
#         self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
#         self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)
#
#
# class QNet(BaseQNet):
#     def __init__(self, dims: [int], state_dim: int, action_dim: int):
#         super().__init__(state_dim=state_dim, action_dim=action_dim)
#         self.net = build_mlp(dims=[state_dim, *dims, action_dim])
#         init_with_orthogonal(self.net[-1], std=0.1)
#
#     def forward(self, state):
#         state = self.state_norm(state)
#         value = self.net(state)
#         value = self.value_re_norm(value)
#         return value  # Q values for multiple actions
#
#     def get_action(self, state):
#         state = self.state_norm(state)
#         if self.explore_rate < torch.rand(1):
#             action = self.net(state).argmax(dim=1, keepdim=True)
#         else:
#             action = torch.randint(self.action_dim, size=(state.shape[0], 1))
#         return action
#
#
# class QNetDuel(BaseQNet):  # Dueling DQN
#     def __init__(self, dims: [int], state_dim: int, action_dim: int):
#         super().__init__(state_dim=state_dim, action_dim=action_dim)
#         self.net_state = build_mlp(dims=[state_dim, *dims])
#         self.net_adv = build_mlp(dims=[dims[-1], 1])  # advantage value
#         self.net_val = build_mlp(dims=[dims[-1], action_dim])  # Q value
#
#         init_with_orthogonal(self.net_adv[-1], std=0.1)
#         init_with_orthogonal(self.net_val[-1], std=0.1)
#
#     def forward(self, state):
#         state = self.state_norm(state)
#         s_enc = self.net_state(state)  # encoded state
#         q_val = self.net_val(s_enc)  # q value
#         q_adv = self.net_adv(s_enc)  # advantage value
#         value = q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # dueling Q value
#         value = self.value_re_norm(value)
#         return value
#
#     def get_action(self, state):
#         state = self.state_norm(state)
#         if self.explore_rate < torch.rand(1):
#             s_enc = self.net_state(state)  # encoded state
#             q_val = self.net_val(s_enc)  # q value
#             action = q_val.argmax(dim=1, keepdim=True)
#         else:
#             action = torch.randint(self.action_dim, size=(state.shape[0], 1))
#         return action
#
#
# class QNetTwin(BaseQNet):  # Double DQN
#     def __init__(self, dims: [int], state_dim: int, action_dim: int):
#         super().__init__(state_dim=state_dim, action_dim=action_dim)
#         self.net_state = build_mlp(dims=[state_dim, *dims])
#         self.net_val1 = build_mlp(dims=[dims[-1], action_dim])  # Q value 1
#         self.net_val2 = build_mlp(dims=[dims[-1], action_dim])  # Q value 2
#         self.soft_max = nn.Softmax(dim=1)
#
#         init_with_orthogonal(self.net_val1[-1], std=0.1)
#         init_with_orthogonal(self.net_val2[-1], std=0.1)
#
#     def forward(self, state):
#         state = self.state_norm(state)
#         s_enc = self.net_state(state)  # encoded state
#         q_val = self.net_val1(s_enc)  # q value
#         return q_val  # one group of Q values
#
#     def get_q1_q2(self, state):
#         state = self.state_norm(state)
#         s_enc = self.net_state(state)  # encoded state
#         q_val1 = self.net_val1(s_enc)  # q value 1
#         q_val1 = self.value_re_norm(q_val1)
#         q_val2 = self.net_val2(s_enc)  # q value 2
#         q_val2 = self.value_re_norm(q_val2)
#         return q_val1, q_val2  # two groups of Q values
#
#     def get_action(self, state):
#         state = self.state_norm(state)
#         s_enc = self.net_state(state)  # encoded state
#         q_val = self.net_val1(s_enc)  # q value
#         if self.explore_rate < torch.rand(1):
#             action = q_val.argmax(dim=1, keepdim=True)
#         else:
#             a_prob = self.soft_max(q_val)
#             action = torch.multinomial(a_prob, num_samples=1)
#         return action
#
#
# class QNetTwinDuel(BaseQNet):  # D3QN: Dueling Double DQN
#     def __init__(self, dims: [int], state_dim: int, action_dim: int):
#         super().__init__(state_dim=state_dim, action_dim=action_dim)
#         self.net_state = build_mlp(dims=[state_dim, *dims])
#         self.net_adv1 = build_mlp(dims=[dims[-1], 1])  # advantage value 1
#         self.net_val1 = build_mlp(dims=[dims[-1], action_dim])  # Q value 1
#         self.net_adv2 = build_mlp(dims=[dims[-1], 1])  # advantage value 2
#         self.net_val2 = build_mlp(dims=[dims[-1], action_dim])  # Q value 2
#         self.soft_max = nn.Softmax(dim=1)
#
#         init_with_orthogonal(self.net_adv1[-1], std=0.1)
#         init_with_orthogonal(self.net_val1[-1], std=0.1)
#         init_with_orthogonal(self.net_adv2[-1], std=0.1)
#         init_with_orthogonal(self.net_val2[-1], std=0.1)
#
#     def forward(self, state):
#         state = self.state_norm(state)
#         s_enc = self.net_state(state)  # encoded state
#         q_val = self.net_val1(s_enc)  # q value
#         q_adv = self.net_adv1(s_enc)  # advantage value
#         value = q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # one dueling Q value
#         value = self.value_re_norm(value)
#         return value
#
#     def get_q1_q2(self, state):
#         state = self.state_norm(state)
#         s_enc = self.net_state(state)  # encoded state
#
#         q_val1 = self.net_val1(s_enc)  # q value 1
#         q_adv1 = self.net_adv1(s_enc)  # advantage value 1
#         q_duel1 = q_val1 - q_val1.mean(dim=1, keepdim=True) + q_adv1
#         q_duel1 = self.value_re_norm(q_duel1)
#
#         q_val2 = self.net_val2(s_enc)  # q value 2
#         q_adv2 = self.net_adv2(s_enc)  # advantage value 2
#         q_duel2 = q_val2 - q_val2.mean(dim=1, keepdim=True) + q_adv2
#         q_duel2 = self.value_re_norm(q_duel2)
#         return q_duel1, q_duel2  # two dueling Q values
#
#     def get_action(self, state):
#         state = self.state_norm(state)
#         s_enc = self.net_state(state)  # encoded state
#         q_val = self.net_val1(s_enc)  # q value
#         if self.explore_rate < torch.rand(1):
#             action = q_val.argmax(dim=1, keepdim=True)
#         else:
#             a_prob = self.soft_max(q_val)
#             action = torch.multinomial(a_prob, num_samples=1)
#         return action
