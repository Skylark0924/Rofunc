"""
Tracking the trajectory with multiple joints by CURI
============================================================

This example runs a Tai Chi demo bimanual trajectory by using CURI.
"""

import numpy as np

import rofunc as rf

traj_l = np.load('../data/LQT_LQR/taichi_1l.npy')
traj_r = np.load('../data/LQT_LQR/taichi_1r.npy')
traj_l[:, 0] += 0.2
traj_r[:, 0] += 0.2
traj_l[:, 1] = -traj_l[:, 1]
traj_r[:, 1] = -traj_r[:, 1]
traj_l[:, 3:] = [0.707, 0., 0, 0.707]
traj_r[:, 3:] = [-0., 0.707, 0.707, -0.]
rf.lqt.plot_3d_bi(traj_l, traj_r, ori=False)

args = rf.config.get_sim_config("CURI")
CURIsim = rf.sim.CURISim(args)
CURIsim.run_traj(traj=[traj_l, traj_r], update_freq=0.001)
