"""
Bimanual iLQR
=============

This example shows how to use the iLQR algorithm to track a bimanual trajectory.
"""

import rofunc as rf
import numpy as np
from rofunc.config.utils import get_config

cfg = get_config('./planning', 'ilqr_bi')

Mu = np.array([[-1, -1.5, 4, 2]]).T  # Target
MuCoM = np.array([0, 1.4])

u0 = np.zeros(cfg.nbVarU * (cfg.nbData - 1))  # Initial control command
x0 = np.array([np.pi / 2, np.pi / 2, np.pi / 3, -np.pi / 2, -np.pi / 3])  # Initial pose

rf.lqr.uni_bi(Mu, MuCoM, u0, x0, cfg)
