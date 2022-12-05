from __future__ import absolute_import

from rofunc.lfd.rl.online import PPOAgent
from rofunc.lfd.rl.offline import CQL
from rofunc.lfd.rl.offline import CRR

algo_map = {
    # "DQN": DQN,
    "PPO": PPOAgent,
    # "SAC": SAC,
    "CQL": CQL,
    "CRR": CRR
}