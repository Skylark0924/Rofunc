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

import os
import random
import sys
import time
from typing import Optional

import cv2
import numpy as np
import torch


def set_seed(seed: Optional[int] = None, deterministic: bool = False) -> int:
    """
    Set the seed for the random number generators
    :param seed: The seed to set. Is None, a random seed will be generated (default: ``None``)
    :param deterministic: Whether PyTorch is configured to use deterministic algorithms (default: ``False``).
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
    cv2.setRNGSeed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # On CUDA 10.1, set environment variable CUDA_LAUNCH_BLOCKING=1
        # On CUDA 10.2 or later, set environment variable CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:2

        # logger.warning("PyTorch/cuDNN deterministic algorithms are enabled. This may affect performance")

    return seed


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_device(v, device) for v in x]
    else:
        return x
