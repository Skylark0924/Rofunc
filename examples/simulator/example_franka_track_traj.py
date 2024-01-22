"""
Tracking the trajectory by Franka
============================================================

This example runs a Tai Chi demo trajectory by using Franka.
"""

import isaacgym
import numpy as np

import rofunc as rf

traj = np.load('../data/LQT_LQR/taichi_1l.npy')
traj[:, 0] -= 0.5
traj[:, 1] += 0.2
traj[:, 2] -= 0.6
traj[:, 3:] = [0., 1, 0, 0]
rf.lqt.plot_3d_uni(traj, ori=False)

args = rf.config.get_sim_config("Franka")
frankasim = rf.sim.FrankaSim(args)
frankasim.run_traj(traj)
