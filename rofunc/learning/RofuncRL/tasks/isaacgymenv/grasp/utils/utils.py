import numpy as np
from omegaconf import OmegaConf
import os
import random
import torch


def set_printoptions(precision: int = 2, threshold: int = 10000,
                     edgeitems: int = 16, linewidth: int = 120,
                     suppress: bool = False) -> None:
    """Sets printoptions of NumPy and PyTorch."""
    np.set_printoptions(precision=precision, threshold=threshold,
                        edgeitems=edgeitems, linewidth=linewidth,
                        suppress=suppress)
    torch.set_printoptions(precision=precision, threshold=threshold,
                           edgeitems=edgeitems, linewidth=linewidth)



