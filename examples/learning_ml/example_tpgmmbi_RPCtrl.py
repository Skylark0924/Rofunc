"""
TP-GMMBi with Relative Parameterization in Control
===================================================

This example shows how to use the TP-GMM in bimanual setting with Relative Parameterization in Control.
"""
import numpy as np

import rofunc as rf
from rofunc.utils.datalab.data_generator.bezier import multi_bezier_demos

left_demo_points = np.array([[[0, 0], [-1, 8], [4, 3], [2, 1], [4, 3]],
                             [[0, -2], [-1, 7], [3, 2.5], [2, 1.6], [4, 3]],
                             [[0, -1], [-1, 8], [4, 5.2], [2, 1.1], [4, 3.5]]])
right_demo_points = np.array([[[8, 8], [7, 1], [4, 3], [6, 8], [4, 3]],
                              [[8, 7], [7, 1], [3, 3], [6, 6], [4, 3]],
                              [[8, 8], [7, 1], [4, 5], [6, 8], [4, 3.5]]])
demos_left_x = multi_bezier_demos(left_demo_points)  # (3, 50, 2): 3 demos, each has 50 points
demos_right_x = multi_bezier_demos(right_demo_points)

# --- TP-GMMBi with Relative Parameterization in Control ---
# Define the task parameters
start_xdx_l = [demos_left_x[i][0] for i in range(len(demos_left_x))]  # TODO: change to xdx
end_xdx_l = [demos_left_x[i][-1] for i in range(len(demos_left_x))]
start_xdx_r = [demos_right_x[i][0] for i in range(len(demos_right_x))]
end_xdx_r = [demos_right_x[i][-1] for i in range(len(demos_right_x))]
task_params = {'left': {'frame_origins': [start_xdx_l, end_xdx_l], 'frame_names': ['start', 'end']},
               'right': {'frame_origins': [start_xdx_r, end_xdx_r], 'frame_names': ['start', 'end']}}
# Fit the model
Repr = rf.RofuncML.TPGMM_RPCtrl(demos_left_x, demos_right_x, task_params, plot=True)
model_l, model_r, model_c = Repr.fit()

# Reproductions for the same situations
Repr.reproduce([model_l, model_r, model_c], show_demo_idx=2)

# Reproductions for new situations
start_xdx_l = [np.array([-0.5, 1, 0, 0])]
end_xdx_l = [np.array([5, 4, 0, 0])]
start_xdx_r = [np.array([6.5, 7, 0, 0])]
end_xdx_r = end_xdx_l

Repr.task_params = {"left": {"frame_origins": [start_xdx_l, end_xdx_l], "frame_names": ["start", "end"]},
                    "right": {"frame_origins": [start_xdx_r, end_xdx_r], "frame_names": ["start", "end"]},
                    "relative": {"frame_origins": [[start_xdx_l[0] - start_xdx_r[0]], [end_xdx_l[0] - end_xdx_r[0]]],
                                 "frame_names": ["start", "end"]}}
traj_l, traj_r, _, _ = Repr.generate([model_l, model_r, model_c], ref_demo_idx=1)
