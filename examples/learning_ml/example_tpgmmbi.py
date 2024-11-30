"""
TP-GMM for bimanual setting
=================

This example shows how to use the TP-GMM in bimanual setting (without coordination).
"""
import numpy as np
import rofunc as rf

left_raw_demo = np.load('../data/LFD_ML/LeftHand.npy')
right_raw_demo = np.load('../data/LFD_ML/RightHand.npy')
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
Repr = rf.RofuncML.TPGMMBi(demos_left_x, demos_right_x, task_params, plot=True)
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
