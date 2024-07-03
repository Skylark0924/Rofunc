"""
Tracking the trajectory with multiple rigid bodies by CURI
============================================================

This example runs a Tai Chi demo bimanual trajectory by using CURI.
"""

import isaacgym
import numpy as np

import rofunc as rf

traj_l = np.load('../data/LQT_LQR/taichi_1l.npy')
traj_r = np.load('../data/LQT_LQR/taichi_1r.npy')
traj_l[:, 0] += 0.8
traj_r[:, 0] += 0.8
traj_l[:, 1] = -traj_l[:, 1]
traj_r[:, 1] = -traj_r[:, 1]
traj_l[:, 3:] = [1, 0, 0, 0]
traj_r[:, 3:] = [0, 0, 1, 0]
rf.lqt.plot_3d_bi(traj_l, traj_r, ori=False)

args = rf.config.get_sim_config("CURI")
CURIsim = rf.sim.CURISim(args)
CURIsim.run_traj(traj=[traj_l, traj_r],
                 # attracted_rigid_bodies=["left_qbhand_palm_link", "right_qbhand_palm_link"],
                 update_freq=0.001)
