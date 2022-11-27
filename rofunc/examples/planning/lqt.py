"""
LQT
=====

This example shows how to use the LQT controller to track a trajectory.
"""

import os
import numpy as np
import rofunc as rf

via_points = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LQT_LQR/rolling_pin_1.npy'))
filter_indices = [0, 1, 5, 10, 22, 36]
via_points = via_points[filter_indices]
u_hat, x_hat, mu, idx_slices = rf.lqt.uni(via_points)
rf.lqt.plot_3d_uni(x_hat, mu, idx_slices, ori=False, save=False)
