"""
Tracking the trajectory by Franka
============================================================

This example runs a Tai Chi demo trajectory by using Franka.
"""
import os
import numpy as np
from isaacgym import gymutil
import rofunc as rf

args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

traj = np.load(os.path.join(rf.file.get_rofunc_path(), 'data/taichi_1l.npy'))
rf.lqt.plot_3d_uni(traj, ori=False)

frankasim = rf.sim.FrankaSim(args)
frankasim.run_traj(traj)
