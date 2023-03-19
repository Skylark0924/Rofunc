from __future__ import absolute_import

from rofunc.learning.rl.online import PPOAgent
from rofunc.learning.rl.offline import CQL
from rofunc.learning.rl.offline import CRR

algo_map = {
    # "DQN": DQN,
    "PPO": PPOAgent,
    # "SAC": SAC,
    "CQL": CQL,
    "CRR": CRR
}