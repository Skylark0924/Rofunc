"""
Tracking the trajectory with multiple joints by CURI
============================================================

This example runs a Tai Chi demo bimanual trajectory by using CURI.
"""
import os
import numpy as np
import rofunc as rf
from isaacgym import gymutil

args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

traj_l = np.load(os.path.join(rf.file.get_rofunc_path(), 'data/taichi_1l.npy'))
traj_r = np.load(os.path.join(rf.file.get_rofunc_path(), 'data/taichi_1r.npy'))
rf.lqt.plot_3d_bi(traj_l, traj_r, ori=False)
rf.curi.run_traj_multi_joints(args, traj=[traj_l, traj_r], update_freq=0.001)
