"""
Tracking the trajectory with multiple joints by walker
============================================================

This example runs customized demo bimanual trajectories by using Walker for HOTO project.
"""
import os
import numpy as np
from isaacgym import gymutil
import rofunc as rf
import quaternion

def motion_prior_generation(traj_l, traj_r, if_convert=False):
    # visualize the trajectory
    rf.lqt.plot_3d_bi(traj_l, traj_r, ori=False)

    # convert the trajectory orientation from Xsens to IsaacGym
    if if_convert:
        for row in traj_l:
            # convert position
            row[0] = row[0] + 0.7
            row[1] = row[1]
            row[2] = row[2] - 0.2

            # convert orientation
            traj_l_qua = quaternion.from_float_array([row[6], row[3], row[4], row[5]])
            q1 = quaternion.from_float_array([0.707, 0.0, 0.707, 0.0])
            # q2 = quaternion.from_float_array([0.707, 0.0, 0.0, 0.707])
            # q3 = quaternion.from_float_array([0.707, 0.0, 0.0, -0.707])
            # q4 = quaternion.from_float_array([0.707, 0.0, 0.707, 0.0])
            # q5 = quaternion.from_float_array([0.707, 0.707, 0.0, 0.0])
            # # q6 = quaternion.from_float_array([0, 0.0, 0.0, 1])
            traj_l_qua = traj_l_qua * q1
            row[6] = traj_l_qua.w
            row[3] = traj_l_qua.x
            row[4] = traj_l_qua.y
            row[5] = traj_l_qua.z

        for row in traj_r:
            # convert position
            row[0] = row[0] + 0.7
            row[1] = row[1]
            row[2] = row[2] - 0.2

            # convert orientation
            traj_r_qua = quaternion.from_float_array([row[6], row[3], row[4], row[5]])
            q1 = quaternion.from_float_array([0.707, 0.0, 0.707, 0.0])
            traj_r_qua = traj_r_qua * q1
            row[6] = traj_r_qua.w
            row[3] = traj_r_qua.x
            row[4] = traj_r_qua.y
            row[5] = traj_r_qua.z

    # setup environment args
    args = gymutil.parse_arguments()
    args.use_gpu_pipeline = False

    # run the trajectory
    CURISim = rf.sim.CURISim(args, asset_root="/home/zhuoli/Rofunc/rofunc/simulator/assets", fix_base_link=True)
    CURISim.init()
    CURISim.run_traj(traj=[traj_l, traj_r], update_freq=0.01)


if __name__ == '__main__':

    traj_l = np.load('../data/HOTO/mvnx/New Session-020/segment/14_LeftHand.npy')
    traj_r = np.load('../data/HOTO/mvnx/New Session-020/segment/10_RightHand.npy')

    motion_prior_generation(traj_r, traj_l, if_convert=True)
