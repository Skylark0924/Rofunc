"""
Tracking the trajectory with multiple rigid bodies by Walker
============================================================

This example runs a Tai Chi demo bimanual trajectory by using Walker.
"""

import numpy as np

import rofunc as rf

traj_l = np.load('../data/LQT_LQR/taichi_1l.npy')
traj_r = np.load('../data/LQT_LQR/taichi_1r.npy')
rf.lqt.plot_3d_bi(traj_l, traj_r, ori=False)

args = rf.config.get_sim_config("Baxter")
baxtersim = rf.sim.BaxterSim(args)
baxtersim.run_traj(traj=[traj_l, traj_r], update_freq=0.001)
