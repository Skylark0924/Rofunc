"""
Tracking the trajectory by Gluon
============================================================

This example runs a Tai Chi demo trajectory by using Gluon.
"""
import numpy as np

import rofunc as rf

traj = np.load('../data/LQT_LQR/taichi_1l.npy') * 0.2
rf.lqt.plot_3d_uni(traj, ori=False)

args = rf.config.get_sim_config("Gluon")
gluonsim = rf.sim.GluonSim(args)
gluonsim.run_traj(traj)
