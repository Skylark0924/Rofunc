from typing import Optional

import os
import sys
import time
import torch
import random
import numpy as np


def set_seed(seed: Optional[int] = None, deterministic: bool = False) -> int:
    """
    Set the seed for the random number generators
    :param seed: The seed to set. Is None, a random seed will be generated (default: ``None``)
    :param deterministic: Whether PyTorch is configured to use deterministic algorithms (default: ``False``).
                          The following environment variables should be established for CUDA 10.1 (``CUDA_LAUNCH_BLOCKING=1``)
                          and for CUDA 10.2 or later (``CUBLAS_WORKSPACE_CONFIG=:16:8`` or ``CUBLAS_WORKSPACE_CONFIG=:4096:2``).
                          See PyTorch `Reproducibility <https://pytorch.org/docs/stable/notes/randomness.html>`_ for details
    """
    # generate a random seed
    if seed is None:
        try:
            seed = int.from_bytes(os.urandom(4), byteorder=sys.byteorder)
        except NotImplementedError:
            seed = int(time.time() * 1000)
        seed %= 2 ** 31  # NumPy's legacy seeding seed must be between 0 and 2**32 - 1

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # On CUDA 10.1, set environment variable CUDA_LAUNCH_BLOCKING=1
        # On CUDA 10.2 or later, set environment variable CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:2

        # logger.warning("PyTorch/cuDNN deterministic algorithms are enabled. This may affect performance")

    return seed
