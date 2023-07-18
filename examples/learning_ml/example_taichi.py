import numpy as np
import rofunc as rf


def uni():
    raw_demo = np.load('/home/lee/Xsens_data/20230310_TaiChi/xsens data_mvnx/010-010/segment/14_LeftHand.npy')
    demos_x = [raw_demo[500:635, :], raw_demo[635:770, :], raw_demo[770:905, :]]

    # --- TP-GMM ---
    # Define the task parameters
    start_xdx = [demos_x[i][0] for i in range(len(demos_x))]  # TODO: change to xdx
    end_xdx = [demos_x[i][-1] for i in range(len(demos_x))]
    task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
    # Fit the model
    Repr = rf.ml.TPGMM(demos_x, task_params, plot=True)
    model = Repr.fit()

    # Reproductions for the same situations
    traj, _ = Repr.reproduce(model, show_demo_idx=2)

    # Reproductions for new situations: set the endpoint as the start point to make a cycled motion
    ref_demo_idx = 2
    start_xdx = [Repr.demos_xdx[ref_demo_idx][0]]
    end_xdx = [Repr.demos_xdx[ref_demo_idx][0]]
    Repr.task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
    traj, _ = Repr.generate(model, ref_demo_idx)

    via_points = traj[:, :7]
    filter_indices = [i for i in range(0, len(via_points), 5)] + [0]
    via_points = via_points[filter_indices]

    controller = rf.planning_control.lqt.LQT(via_points)
    u_hat, x_hat, mu, idx_slices = controller.solve()
    rf.lqt.plot_3d_uni(x_hat, mu, idx_slices, ori=False, save=True, save_file_name='v_right.npy')


def uni_Imp():
    raw_demo = np.load('/home/lee/Xsens_data/20230310_TaiChi/xsens data_mvnx/010-010/segment/14_LeftHand.npy')
    raw_demo_imp = np.array(np.loadtxt('/home/lee/Xsens_data/20230310_TaiChi/stiffness estimation/stiffness_test.txt'))
    raw_demo_imp = np.reshape(raw_demo_imp, (-1, 1))
    raw_demo_imp = np.tile(raw_demo_imp, 2)
    demos_x = [raw_demo[500:635, :], raw_demo[635:770, :], raw_demo[770:905, :]]
    demos_imp = [raw_demo_imp[:99], raw_demo_imp[100:199], raw_demo_imp[200:299]]
    # --- TP-GMM ---
    # Define the task parameters
    start_xdx = [demos_x[i][0] for i in range(len(demos_x))]  # TODO: change to xdx
    end_xdx = [demos_x[i][-1] for i in range(len(demos_x))]
    task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}

    # Fit the model
    Repr = rf.ml.TPGMM(demos_x, task_params, plot=True)
    model = Repr.fit()

    # Reproductions for the same situations
    traj, _ = Repr.reproduce(model, show_demo_idx=2)

    # --- TP-GMM (Impedance) ---
    start_xdx_imp = [demos_imp[i][0] for i in range(len(demos_imp))]
    end_xdx_imp = [demos_imp[i][-1] for i in range(len(demos_imp))]
    task_params_imp = {'frame_origins': [start_xdx_imp, end_xdx_imp], 'frame_names': ['start', 'end']}

    Repr_imp = rf.ml.TPGMM(demos_imp, task_params_imp, plot=True)
    model_imp = Repr_imp.fit()

    imp, _ = Repr_imp.reproduce(model_imp, show_demo_idx=2)

    # Reproductions for new situations: set the endpoint as the start point to make a cycled motion
    ref_demo_idx = 2
    start_xdx = [Repr.demos_xdx[ref_demo_idx][0]]
    end_xdx = [Repr.demos_xdx[ref_demo_idx][0]]
    Repr.task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
    traj, _ = Repr.generate(model, ref_demo_idx)

    via_points = traj[:, :7]
    filter_indices = [i for i in range(0, len(via_points), 5)] + [0]
    via_points = via_points[filter_indices]

    controller = rf.planning_control.lqt.LQT(via_points)
    u_hat, x_hat, mu, idx_slices = controller.solve()
    rf.lqt.plot_3d_uni(x_hat, mu, idx_slices, ori=False, save=True, save_file_name=False)


def bi():
    left_raw_demo = np.load('/home/lee/Xsens_data/20230310_TaiChi/xsens data_mvnx/010-010/segment/14_LeftHand.npy')
    right_raw_demo = np.load('/home/lee/Xsens_data/20230310_TaiChi/xsens data_mvnx/010-010/segment/10_RightHand.npy')
    demos_left_x = [left_raw_demo[500:635, :], left_raw_demo[635:770, :], left_raw_demo[770:905, :]]
    demos_right_x = [right_raw_demo[500:635, :], right_raw_demo[635:770, :], right_raw_demo[770:905, :]]

    # --- TP-GMMBi ---
    # Define the task parameters
    start_xdx_l = [demos_left_x[i][0] for i in range(len(demos_left_x))]  # TODO: change to xdx
    end_xdx_l = [demos_left_x[i][-1] for i in range(len(demos_left_x))]
    start_xdx_r = [demos_right_x[i][0] for i in range(len(demos_right_x))]
    end_xdx_r = [demos_right_x[i][-1] for i in range(len(demos_right_x))]
    task_params = {'left': {'frame_origins': [start_xdx_l, end_xdx_l], 'frame_names': ['start', 'end']},
                   'right': {'frame_origins': [start_xdx_r, end_xdx_r], 'frame_names': ['start', 'end']}}
    # Fit the model
    Repr = rf.ml.TPGMMBi(demos_left_x, demos_right_x, task_params, plot=True)
    model_l, model_r = Repr.fit()

    # Reproductions for the same situations
    traj_l, traj_r, _, _ = Repr.reproduce([model_l, model_r], show_demo_idx=2)

    # Reproductions for new situations
    ref_demo_idx = 2
    start_xdx_l = [Repr.repr_l.demos_xdx[ref_demo_idx][0]]
    end_xdx_l = [Repr.repr_l.demos_xdx[ref_demo_idx][0]]
    start_xdx_r = [Repr.repr_r.demos_xdx[ref_demo_idx][0]]
    end_xdx_r = [Repr.repr_r.demos_xdx[ref_demo_idx][0]]
    Repr.task_params = {'left': {'frame_origins': [start_xdx_l, end_xdx_l], 'frame_names': ['start', 'end']},
                        'right': {'frame_origins': [start_xdx_r, end_xdx_r], 'frame_names': ['start', 'end']}}
    traj_l, traj_r, _, _ = Repr.generate([model_l, model_r], ref_demo_idx)

    via_points = traj_l[:, :7]
    filter_indices = [i for i in range(0, len(via_points) - 5, 5)] + [0]
    via_points = via_points[filter_indices]

    controller = rf.planning_control.lqt.LQT(via_points)
    u_hat, x_hat_l, mu_l, idx_slices = controller.solve()

    via_points = traj_r[:, :7]
    filter_indices = [i for i in range(0, len(via_points) - 5, 5)] + [0]
    via_points = via_points[filter_indices]

    controller = rf.planning_control.lqt.LQT(via_points)
    u_hat, x_hat_r, mu_r, idx_slices = controller.solve()
    rf.lqt.plot_3d_bi(x_hat_l, x_hat_r, mu_l, mu_r, idx_slices, ori=False, save=True,
                      save_file_name=['h_left.npy', 'h_right.npy'])


def export():
    # mvnx_file = '/home/ubuntu/Downloads/OneDrive_2023-03-10/010-004.mvnx'
    mvnx_file = '/home/lee/Xsens_data/20230310_TaiChi/xsens data_mvnx/010-010.mvnx'
    rf.xsens.export(mvnx_file)


if __name__ == '__main__':
    uni_Imp()
