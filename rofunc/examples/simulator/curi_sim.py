"""
CURI Simulator
=================================

This example runs a Tai Chi demo bimanual trajectory by using CURI.
"""

import numpy as np
from importlib_resources import files
import rofunc as rf
from isaacgym import gymutil

args = gymutil.parse_arguments()

traj_l = np.load(files('rofunc.data').joinpath('taichi_1l.npy'))
traj_r = np.load(files('rofunc.data').joinpath('taichi_1r.npy'))
rf.lqt.plot_3d_bi(traj_l, traj_r, ori=False)
rf.curi.run_traj_bi(args, traj_l, traj_r, update_freq=0.001)
