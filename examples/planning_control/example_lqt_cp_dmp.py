"""
LQT with control primitives and DMP
==================================

This example shows how to use the LQT controller with control primitives and DMP to track a trajectory.
"""

import numpy as np
import rofunc as rf
from scipy.interpolate import interp1d

cfg = rf.config.utils.get_config('./planning', 'lqt_cp_dmp')
cfg.nbDeriv = 3
x = np.load('../data/LQT_LQR/S.npy')[0, :, :2].T

f_pos = interp1d(np.linspace(0, np.size(x, 1) - 1, np.size(x, 1), dtype=int), x, kind='cubic')
MuPos = f_pos(np.linspace(0, np.size(x, 1) - 1, cfg.nbData))  # Position
MuVel = np.gradient(MuPos)[1] / cfg.dt
MuAcc = np.gradient(MuVel)[1] / cfg.dt
# Position, velocity and acceleration profiles as references
via_points = np.vstack((MuPos, MuVel, MuAcc))
cfg.nbData = len(via_points[0])

controller = rf.planning_control.lqt.lqt_cp_dmp.LQTCPDMP(via_points, cfg)
# u_hat, x_hat, mu, idx_slices = controller.solve()
# rf.lqt.plot_3d_uni([x_hat], mu, idx_slices)
# rf.lqt.uni_cp_dmp(via_points, cfg)
