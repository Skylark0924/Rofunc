"""
Tracking the trajectory with multiple joints by CURI
============================================================

This example runs a Tai Chi demo bimanual trajectory by using CURI.
"""
import os
import numpy as np
from isaacgym import gymutil
import rofunc as rf

args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

traj_l = np.load(os.path.join(rf.file.get_rofunc_path(), 'data/taichi_1l.npy'))
traj_r = np.load(os.path.join(rf.file.get_rofunc_path(), 'data/taichi_1r.npy'))
rf.lqt.plot_3d_bi(traj_l, traj_r, ori=False)

CURIsim = rf.sim.CURISim(args)
CURIsim.run_traj(traj=[traj_l, traj_r], update_freq=0.001)
