import torch


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, device):  # shape:the dimension of input data
        self.n = 0
        self.mean = torch.zeros(shape).to(device)
        self.S = torch.zeros(shape).to(device)
        self.std = torch.sqrt(self.S).to(device)

    def update(self, x):
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

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
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


def empty_preprocessor(_input):
    return _input
