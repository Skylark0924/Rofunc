"""
iLQR control primitive version
==============================

This example shows how to use the iLQR solver with control primitive.
"""

import rofunc as rf
import numpy as np
from rofunc.config.utils import get_config

cfg = get_config('./planning', 'ilqr')

# Via-points
Mu = np.array([[2, 1, -np.pi / 2], [3, 1, -np.pi / 2]])  # Via-points
Rot = np.zeros([2, 2, cfg.nbPoints])  # Object orientation matrices

# Object rotation matrices
for t in range(cfg.nbPoints):
    orn_t = Mu[t, -1]
    Rot[t] = np.asarray([
        [np.cos(orn_t), -np.sin(orn_t)],
        [np.sin(orn_t), np.cos(orn_t)]
    ])

u0 = np.zeros(cfg.nbVarU * (cfg.nbData - 1))  # Initial control command
x0 = np.array([3 * np.pi / 4, -np.pi / 2, -np.pi / 4])  # Initial state

rf.lqr.uni_cp(Mu, Rot, u0, x0, cfg)
