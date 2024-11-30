import numpy as np

import rofunc as rf
from rofunc.utils.datalab.data_generator.bezier import multi_bezier_demos
from rofunc.utils.datalab.data_generator import draw_arc


def test_bi_spatial_data():
    # Create demos for dual arms via bezier
    left_demo_points = np.array([[[0, 0], [-1, 8], [4, 3], [2, 1], [4, 3]],
                                 [[0, -2], [-1, 7], [3, 2.5], [2, 1.6], [4, 3]],
                                 [[0, -1], [-1, 8], [4, 5.2], [2, 1.1], [4, 3.5]]])
    right_demo_points = np.array([[[8, 8], [7, 1], [4, 3], [6, 8], [4, 3]],
                                  [[8, 7], [7, 1], [3, 3], [6, 6], [4, 3]],
                                  [[8, 8], [7, 1], [4, 5], [6, 8], [4, 3.5]]])
    demos_left_x = multi_bezier_demos(left_demo_points)  # (3, 50, 2): 3 demos, each has 50 points
    demos_right_x = multi_bezier_demos(right_demo_points)

    start_xdx_l = [demos_left_x[i][0] for i in range(len(demos_left_x))]  # TODO: change to xdx
    end_xdx_l = [demos_left_x[i][-1] for i in range(len(demos_left_x))]
    start_xdx_r = [demos_right_x[i][0] for i in range(len(demos_right_x))]
    end_xdx_r = [demos_right_x[i][-1] for i in range(len(demos_right_x))]
    task_params = {'left': {'frame_origins': [start_xdx_l, end_xdx_l], 'frame_names': ['start', 'end']},
                   'right': {'frame_origins': [start_xdx_r, end_xdx_r], 'frame_names': ['start', 'end']}}

    Repr = rf.RofuncML.TPGMM_RPCtrl(demos_left_x, demos_right_x, task_params, plot=False)
    model_l, model_r, model_c = Repr.fit()

    Repr.reproduce([model_l, model_r, model_c], show_demo_idx=2)


def test_bi_temporal_data():
    # Create demos for dual arms via bezier
    theta_right = np.array([[-1 * np.pi / 4, 1 * np.pi / 5],
                            [-2 * np.pi / 4, 1 * np.pi / 6],
                            [-1 * np.pi / 4, 2 * np.pi / 7],
                            [-1 * np.pi / 4, 1 * np.pi / 7]])
    theta_left = np.pi + theta_right

    demos_left_x = np.vstack((draw_arc([-1, 0], 1, theta_left[0, 0], theta_left[0, 1], 'orange'),
                              draw_arc([2, -3], 2, theta_left[1, 0], theta_left[1, 1], 'purple'),
                              draw_arc([1, -2], 3, theta_left[2, 0], theta_left[2, 1], 'grey'),
                              draw_arc([1, -2], 2.5, theta_left[3, 0], theta_left[3, 1],
                                       'blue')))  # (3, 100, 2): 3 demos, each has 100 points
    demos_right_x = np.vstack((draw_arc([-1, 0], 2, theta_right[0, 0], theta_right[0, 1], 'orange'),
                               draw_arc([2, -3], 3, theta_right[1, 0], theta_right[1, 1], 'purple'),
                               draw_arc([1, -2], 4, theta_right[2, 0], theta_right[2, 1], 'grey'),
                               draw_arc([1, -2], 1, theta_right[3, 0], theta_right[3, 1], 'blue')))

    start_xdx_l = [demos_left_x[i][0] for i in range(len(demos_left_x))]  # TODO: change to xdx
    end_xdx_l = [demos_left_x[i][-1] for i in range(len(demos_left_x))]
    start_xdx_r = [demos_right_x[i][0] for i in range(len(demos_right_x))]
    end_xdx_r = [demos_right_x[i][-1] for i in range(len(demos_right_x))]
    task_params = {'left': {'frame_origins': [start_xdx_l, end_xdx_l], 'frame_names': ['start', 'end']},
                   'right': {'frame_origins': [start_xdx_r, end_xdx_r], 'frame_names': ['start', 'end']}}

    Repr = rf.RofuncML.TPGMM_RPCtrl(demos_left_x, demos_right_x, task_params, plot=False)
    model_l, model_r, model_c = Repr.fit()

    Repr.reproduce([model_l, model_r, model_c], show_demo_idx=2)


if __name__ == '__main__':
    test_bi_spatial_data()
    test_bi_temporal_data()
