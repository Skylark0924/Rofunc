"""
Tracking the trajectory by Franka
============================================================

This example runs a Tai Chi demo trajectory by using Franka.
"""

import numpy as np

import rofunc as rf

traj = np.load('../data/LQT_LQR/taichi_1l.npy')
rf.lqt.plot_3d_uni(traj, ori=False)

args = rf.config.get_sim_config("Franka")
frankasim = rf.sim.FrankaSim(args)
frankasim.run_traj(traj)
