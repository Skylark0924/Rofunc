import numpy as np

import rofunc as rf


def test_2d_uni_tpgmm():
    demo_points = np.array([[[0, 0], [-1, 8], [4, 3], [2, 1], [4, 3]],
                            [[0, -2], [-1, 7], [3, 2.5], [2, 1.6], [4, 3]],
                            [[0, -1], [-1, 8], [4, 5.2], [2, 1.1], [4, 3.5]]])
    demos_x = rf.data_generator.multi_bezier_demos(demo_points)  # (3, 50, 2): 3 demos, each has 50 points

    start_xdx = [demos_x[i][0] for i in range(len(demos_x))]  # TODO: change to xdx
    end_xdx = [demos_x[i][-1] for i in range(len(demos_x))]
    task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
    Repr = rf.RofuncML.TPGMM(demos_x, task_params, plot=False)
    model = Repr.fit()

    # Reproductions for the same situations
    traj, _ = Repr.reproduce(model, show_demo_idx=2)


def test_2d_bi_tpgmm():
    left_demo_points = np.array([[[0, 0], [-1, 8], [4, 3], [2, 1], [4, 3]],
                                 [[0, -2], [-1, 7], [3, 2.5], [2, 1.6], [4, 3]],
                                 [[0, -1], [-1, 8], [4, 5.2], [2, 1.1], [4, 3.5]]])
    right_demo_points = np.array([[[8, 8], [7, 1], [4, 3], [6, 8], [4, 3]],
                                  [[8, 7], [7, 1], [3, 3], [6, 6], [4, 3]],
                                  [[8, 8], [7, 1], [4, 5], [6, 8], [4, 3.5]]])
    demos_left_x = rf.data_generator.multi_bezier_demos(left_demo_points)  # (3, 50, 2): 3 demos, each has 50 points
    demos_right_x = rf.data_generator.multi_bezier_demos(right_demo_points)

    # Define the task parameters
    start_xdx_l = [demos_left_x[i][0] for i in range(len(demos_left_x))]  # TODO: change to xdx
    end_xdx_l = [demos_left_x[i][-1] for i in range(len(demos_left_x))]
    start_xdx_r = [demos_right_x[i][0] for i in range(len(demos_right_x))]
    end_xdx_r = [demos_right_x[i][-1] for i in range(len(demos_right_x))]
    task_params = {'left': {'frame_origins': [start_xdx_l, end_xdx_l], 'frame_names': ['start', 'end']},
                   'right': {'frame_origins': [start_xdx_r, end_xdx_r], 'frame_names': ['start', 'end']}}
    Repr = rf.RofuncML.TPGMMBi(demos_left_x, demos_right_x, task_params, plot=False)
    model_l, model_r = Repr.fit()

    traj_l, traj_r, _, _ = Repr.reproduce([model_l, model_r], show_demo_idx=2)


# def test_7d_uni_tpgmm():
#     raw_demo = np.load('../examples/data/LFD_ML/LeftHand.npy')
#     demos_x = [raw_demo[500:635, :], raw_demo[635:770, :], raw_demo[770:905, :]]
#
#     start_xdx = [demos_x[i][0] for i in range(len(demos_x))]  # TODO: change to xdx
#     end_xdx = [demos_x[i][-1] for i in range(len(demos_x))]
#     task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
#     Repr = rf.RofuncML.TPGMM(demos_x, task_params, plot=False)
#     model = Repr.fit()
#
#     # Reproductions for the same situations
#     traj, _ = Repr.reproduce(model, show_demo_idx=2)
#
#     # Reproductions for new situations
#     ref_demo_idx = 2
#     start_xdx = [Repr.demos_xdx[ref_demo_idx][0]]
#     end_xdx = [Repr.demos_xdx[ref_demo_idx][0]]
#     Repr.task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
#     traj, _ = Repr.generate(model, ref_demo_idx)
#
#
# def test_7d_bi_tpgmm():
#     left_raw_demo = np.load('../examples/data/LFD_ML/LeftHand.npy')
#     right_raw_demo = np.load('../examples/data/LFD_ML/RightHand.npy')
#     demos_left_x = [left_raw_demo[500:635, :], left_raw_demo[635:770, :], left_raw_demo[770:905, :]]
#     demos_right_x = [right_raw_demo[500:635, :], right_raw_demo[635:770, :], right_raw_demo[770:905, :]]
#
#     # --- TP-GMMBi ---
#     # Define the task parameters
#     start_xdx_l = [demos_left_x[i][0] for i in range(len(demos_left_x))]  # TODO: change to xdx
#     end_xdx_l = [demos_left_x[i][-1] for i in range(len(demos_left_x))]
#     start_xdx_r = [demos_right_x[i][0] for i in range(len(demos_right_x))]
#     end_xdx_r = [demos_right_x[i][-1] for i in range(len(demos_right_x))]
#     task_params = {'left': {'frame_origins': [start_xdx_l, end_xdx_l], 'frame_names': ['start', 'end']},
#                    'right': {'frame_origins': [start_xdx_r, end_xdx_r], 'frame_names': ['start', 'end']}}
#     # Fit the model
#     Repr = rf.RofuncML.TPGMMBi(demos_left_x, demos_right_x, task_params, plot=False)
#     model_l, model_r = Repr.fit()
#
#     # Reproductions for the same situations
#     traj_l, traj_r, _, _ = Repr.reproduce([model_l, model_r], show_demo_idx=2)
#
#     # Reproductions for new situations
#     ref_demo_idx = 2
#     start_xdx_l = [Repr.repr_l.demos_xdx[ref_demo_idx][0]]
#     end_xdx_l = [Repr.repr_l.demos_xdx[ref_demo_idx][0]]
#     start_xdx_r = [Repr.repr_r.demos_xdx[ref_demo_idx][0]]
#     end_xdx_r = [Repr.repr_r.demos_xdx[ref_demo_idx][0]]
#     Repr.task_params = {'left': {'frame_origins': [start_xdx_l, end_xdx_l], 'frame_names': ['start', 'end']},
#                         'right': {'frame_origins': [start_xdx_r, end_xdx_r], 'frame_names': ['start', 'end']}}
#     traj_l, traj_r, _, _ = Repr.generate([model_l, model_r], ref_demo_idx)


if __name__ == '__main__':
    # test_7d_uni_tpgmm()
    # test_7d_bi_tpgmm()
    test_2d_uni_tpgmm()
    test_2d_bi_tpgmm()
