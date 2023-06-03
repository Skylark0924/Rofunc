import time
from rofunc.learning.rl.trainers.base_trainer import BaseTrainer


class PPOTrainer(BaseTrainer):
    def __init__(self, cfg, env, device):
        super().__init__(cfg, env, device)
        pass

    def train_episode(self):
        pass

    def eval(self):
        pass

    def inference(self):
        pass
