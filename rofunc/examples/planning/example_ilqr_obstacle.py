"""
iLQR with obstacle avoidance
==============================

This example shows how to use the iLQR solver with obstacle avoidance.
"""

import rofunc as rf
import numpy as np
from rofunc.config.utils import get_config

cfg = get_config('./planning', 'ilqr_obstacle')

Mu = np.array([[3, 3, np.pi / 6]])  # Via-point [x1,x2,o]
Obst = np.array([
    [1, 0.6, np.pi / 4],  # [x1,x2,o]
    [2, 2.5, -np.pi / 6]  # [x1,x2,o]
])

A_obst = np.zeros((cfg.nbObstacles, 2, 2))
S_obst = np.zeros((cfg.nbObstacles, 2, 2))
Q_obst = np.zeros((cfg.nbObstacles, 2, 2))
U_obst = np.zeros((cfg.nbObstacles, 2, 2))  # Q_obs[t] = U_obs[t].T @ U_obs[t]
for i in range(cfg.nbObstacles):
    orn_t = Obst[i][-1]
    A_obst[i] = np.array([  # Orientation in matrix form
        [np.cos(orn_t), -np.sin(orn_t)],
        [np.sin(orn_t), np.cos(orn_t)]
    ])

    S_obst[i] = A_obst[i] @ np.diag(cfg.sizeObstacle) ** 2 @ A_obst[i].T  # Covariance matrix
    Q_obst[i] = np.linalg.inv(S_obst[i])  # Precision matrix
    U_obst[i] = A_obst[i] @ np.diag(
        1 / np.array(cfg.sizeObstacle))  # "Square root" of cfg.Q_obst[i]

u0 = np.zeros(cfg.nbVarU * (cfg.nbData - 1))  # Initial control command
x0 = np.zeros(cfg.nbVarX)  # Initial state

rf.lqr.uni_obstacle(Mu, Obst, S_obst, U_obst, u0, x0, cfg)
