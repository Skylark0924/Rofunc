"""
Tracking the trajectory with multiple joints by CURI
============================================================

This example runs a Tai Chi demo bimanual trajectory by using Walker.
"""
import os
import numpy as np
from isaacgym import gymutil
import rofunc as rf
import quaternion


args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

traj_l = np.load('../data/HOTO/mvnx/New Session-012/segment/14_LeftHand.npy')
traj_r = np.load('../data/HOTO/mvnx/New Session-012/segment/10_RightHand.npy')
rf.lqt.plot_3d_bi(traj_l, traj_r, ori=False)

for row in traj_l:
    q1 = quaternion.from_float_array([row[6], row[3], row[4], row[5]])
    q2 = quaternion.from_float_array([0.0, 0.0, 1.0, 0.0])
    q4 = quaternion.from_float_array([0.707, 0.0, 0.0, -0.707])
    q3 = q1 * q2 * q4
    row[6] = q3.w
    row[3] = q3.x
    row[4] = q3.y
    row[5] = q3.z

# traj = [traj_l, traj_r]
# traj_array = np.array(traj)
# print(traj_array.shape)

Walkersim = rf.sim.WalkerSim(args, asset_root="/home/zhuoli/Rofunc/rofunc/simulator/assets", fix_base_link=True)
Walkersim.init()
Walkersim.run_traj(traj=[traj_l, traj_r], update_freq=0.1)
