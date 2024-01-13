import numpy as np
from omegaconf import OmegaConf
import os
import random
import torch


def register_omegaconf_resolvers() -> None:
    """Registers custom resolvers for OmegaConf."""
    OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver(
        'contains', lambda x, y: x.lower() in y.lower())
    OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver(
        'resolve_default', lambda default, arg: default if arg == '' else arg)


def set_seed(seed, torch_deterministic=False):
    """Sets the seed for all used modules.

    Will choose a random seed if seed=-1 and not deterministic.
    """
    if seed == -1 and torch_deterministic:
        seed = 0
    elif seed == -1:
        seed = np.random.randint(0, 10000)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    return seed