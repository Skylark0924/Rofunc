"""
Tracking the trajectory with multiple joints by CURI
============================================================

This example runs a Tai Chi demo bimanual trajectory by using CURI.
"""
import os
import numpy as np
from isaacgym import gymutil
import rofunc as rf


traj_l = np.load('../data/LQT_LQR/taichi_1l.npy')
traj_r = np.load('../data/LQT_LQR/taichi_1r.npy')
rf.lqt.plot_3d_bi(traj_l, traj_r, ori=False)

args = rf.config.get_sim_config("CURI")
CURIsim = rf.sim.CURISim(args)
CURIsim.run_traj(traj=[traj_l, traj_r], update_freq=0.001)
