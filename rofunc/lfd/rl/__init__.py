from __future__ import absolute_import

from rofunc.lfd.rl.online import DQN
from rofunc.lfd.rl.online import PPO
from rofunc.lfd.rl.online import SAC
from rofunc.lfd.rl.offline import CQL
from rofunc.lfd.rl.offline import CRR

algo_map = {
    "DQN": DQN,
    "PPO": PPO,
    "SAC": SAC,
    "CQL": CQL,
    "CRR": CRR
}