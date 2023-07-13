"""
iLQR
====

This example shows how to use the iLQR algorithm to track a trajectory in a 3-dimensional space.
"""

import rofunc as rf
import numpy as np
from scipy.spatial.transform import Rotation as R
from rofunc.config.utils import get_config
from rofunc.planning_control.lqr.ilqr_3d import iLQR_3D

cfg = get_config('./planning', 'ilqr3d')

Mu = np.array([[0.10021176, -0.01036865, 0.49858453, 0.61920622, 0.19450308, 0.21959119, 0.72837622],
               [0.08177911, -0.06516777, 0.44698613, 0.88028369, 0.03904804, 0.02095377, 0.4723736],
               [0.07767701, -0.04641878, 0.4275838, 0.79016704, 0.01637976, 0.01269766, 0.61254103],
               [0.06642697, 0.28006863, 0.39004221, 0.34475831, 0.01169578, 0.01790368, 0.93844785]])

Rot = np.zeros([cfg.nbPoints, 3, 3])  # Object orientation matrices
# Object rotation matrices
for t in range(cfg.nbPoints):
    orn_t = Mu[t, 3:]

    quat = R.from_quat(orn_t)

    Rot[t] = quat.as_matrix()

u0 = np.zeros(cfg.nbVarU * (cfg.nbData - 1))  # Initial control command
x0 = np.array([0, 0, 0, 0, 0, 0])  # Initial state
controller = iLQR_3D(cfg)
u, x = controller.solve(Mu, Rot, u0, x0, for_test=True)
