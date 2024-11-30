"""
Taichi
======

This example shows how to use the TPGMM and TPGMMBi classes to learn Taichi pushing hand on a humanoid robot./
"""

import numpy as np
import rofunc as rf


# Parameters
task_param = {
    "nbData": 500,  # Number of data points
    "nbPoints": 3,  # Number of viapoints
    "nbVarPos": 7,  # Dimension of position data
    "nbDeriv": 2,  # Number of static and dynamic features (2 -> [x,dx])
    "dt": 1e-2,  # Time step duration
    "rfactor": 1e-8  # Control cost
}
task_param["nb_var"] = task_param["nbVarPos"] * task_param["nbDeriv"]


def uni():
    raw_demo = np.load('/home/ubuntu/Downloads/OneDrive_2023-03-10/010-003/LeftHand.npy')
    raw_demo = np.expand_dims(raw_demo, axis=0)
    demos_x = np.vstack((raw_demo[:, 430:525, :], raw_demo[:, 240:335, :], raw_demo[:, 335:430, :]))

    # Reproductions for the same situations
    representation = rf.RofuncML.tpgmm.TPGMM(demos_x)
    model = representation.fit(plot=True)

    # Reproductions for new situations
    ref_demo_idx = 2
    A, b = representation.demos_A_xdx[ref_demo_idx][0], representation.demos_b_xdx[ref_demo_idx][0]
    b[1] = b[0]
    task_params = {'A': A, 'b': b}
    traj = representation.generate(model, ref_demo_idx, task_params, plot=True)

    via_points = traj[:, :7]
    filter_indices = [i for i in range(0, len(via_points), 5)] + [0]
    via_points = via_points[filter_indices]

    controller = rf.planning.lqt.LQT(via_points)
    u_hat, x_hat, mu, idx_slices = controller.solve()
    rf.lqt.plot_3d_uni(x_hat, mu, idx_slices, ori=False, save=True, save_file_name='v_right.npy')


def bi():
    # raw_demo = np.load('/home/ubuntu/Downloads/OneDrive_2023-03-10/010-004/LeftHand.npy')
    raw_demo = np.load('/home/ubuntu/Data/20230704/xsens data_mvnx/010-010/segment/14_LeftHand.npy')
    raw_demo = np.expand_dims(raw_demo, axis=0)
    # demos_left_x = np.vstack((raw_demo[:, 134:254, :], raw_demo[:, 250:370, :], raw_demo[:, 370:490, :]))
    demos_left_x = np.vstack((raw_demo[:, 500:635, :], raw_demo[:, 635:770, :], raw_demo[:, 770:905, :]))
    # raw_demo = np.load('/home/ubuntu/Downloads/OneDrive_2023-03-10/010-004/RightHand.npy')
    raw_demo = np.load('/home/ubuntu/Data/20230704/xsens data_mvnx/010-010/segment/10_RightHand.npy')
    raw_demo = np.expand_dims(raw_demo, axis=0)
    # demos_right_x = np.vstack((raw_demo[:, 134:254, :], raw_demo[:, 250:370, :], raw_demo[:, 370:490, :]))
    demos_right_x = np.vstack((raw_demo[:, 500:635, :], raw_demo[:, 635:770, :], raw_demo[:, 770:905, :]))

    # Define the task parameters
    start_xdx_l = [demos_left_x[i][0] for i in range(len(demos_left_x))]  # TODO: change to xdx
    end_xdx_l = [demos_left_x[i][-1] for i in range(len(demos_left_x))]
    start_xdx_r = [demos_right_x[i][0] for i in range(len(demos_right_x))]
    end_xdx_r = [demos_right_x[i][-1] for i in range(len(demos_right_x))]
    task_params = {'left': {'frame_origins': [start_xdx_l, end_xdx_l], 'frame_names': ['start', 'end']},
                   'right': {'frame_origins': [start_xdx_r, end_xdx_r], 'frame_names': ['start', 'end']}}

    representation = rf.RofuncML.tpgmm.TPGMMBi(demos_left_x, demos_right_x, task_params, plot=True)
    model_l, model_r = representation.fit()

    # # Reproductions for the same situations
    # traj_l, traj_r = representation.reproduce(model_l, model_r, show_demo_idx=2, plot=True)

    # Reproductions for new situations
    ref_demo_idx = 2
    A_l, b_l = representation.repr_l.demos_A_xdx[ref_demo_idx][0], representation.repr_l.demos_b_xdx[ref_demo_idx][0]
    b_l[1] = b_l[0]
    A_r, b_r = representation.repr_r.demos_A_xdx[ref_demo_idx][0], representation.repr_r.demos_b_xdx[ref_demo_idx][0]
    b_r[1] = b_r[0]
    task_params = {'Left': {'A': A_l, 'b': b_l}, 'Right': {'A': A_r, 'b': b_r}}
    traj_l, traj_r = representation.generate(model_l, model_r, ref_demo_idx, task_params, plot=True)

    via_points = traj_l[:, :7]
    filter_indices = [i for i in range(0, len(via_points) - 5, 5)] + [0]
    via_points = via_points[filter_indices]

    controller = rf.planning.lqt.LQT(via_points)
    u_hat, x_hat_l, mu_l, idx_slices = controller.solve()

    via_points = traj_r[:, :7]
    filter_indices = [i for i in range(0, len(via_points) - 5, 5)] + [0]
    via_points = via_points[filter_indices]

    controller = rf.planning.lqt.LQT(via_points)
    u_hat, x_hat_r, mu_r, idx_slices = controller.solve()
    rf.lqt.plot_3d_bi(x_hat_l, x_hat_r, mu_l, mu_r, idx_slices, ori=False, save=True,
                      save_file_name=['h_left.npy', 'h_right.npy'])


def export():
    # mvnx_file = '/home/ubuntu/Downloads/OneDrive_2023-03-10/010-004.mvnx'
    mvnx_file = '/home/ubuntu/Data/20230704/xsens data_mvnx/010-010.mvnx'
    rf.xsens.export(mvnx_file)


if __name__ == '__main__':
   bi()
