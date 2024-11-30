"""
TP-GMR for Tai Chi
=================

This example shows how to use the TP-GMR to learn a Tai Chi movement.
"""
import numpy as np

import rofunc as rf

raw_demo = np.load('../data/LFD_ML/LeftHand.npy')
demos_x = [raw_demo[500:635, :], raw_demo[635:770, :], raw_demo[770:905, :]]

# --- TP-GMR ---
# Define the task parameters
start_xdx = [demos_x[i][0] for i in range(len(demos_x))]  # TODO: change to xdx
end_xdx = [demos_x[i][-1] for i in range(len(demos_x))]
task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
# Fit the model
Repr = rf.RofuncML.TPGMR(demos_x, task_params, plot=True)
model = Repr.fit()

# Reproductions for the same situations
traj, _ = Repr.reproduce(model, show_demo_idx=2)

# Reproductions for new situations
ref_demo_idx = 2
start_xdx = [Repr.demos_xdx[ref_demo_idx][0]]
end_xdx = [Repr.demos_xdx[ref_demo_idx][0]]
Repr.task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
traj, _ = Repr.generate(model, ref_demo_idx, task_params)

# raw_demo = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LFD_ML/LeftHand.npy'))
# demos_left_x = [raw_demo[500:635, :], raw_demo[635:770, :], raw_demo[770:905, :]]
# raw_demo = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LFD_ML/RightHand.npy'))
# demos_right_x = [raw_demo[500:635, :], raw_demo[635:770, :], raw_demo[770:905, :]]
#
# Repr = rf.RofuncML.TPGMRBi(demos_left_x, demos_right_x, horizon=300, plot=True)
# model_l, model_r = Repr.fit()
#
# # Reproductions for the same situations
# traj_l, traj_r = Repr.reproduce(model_l, model_r, show_demo_idx=2)
#
# # Reproductions for new situations
# ref_demo_idx = 2
# start_xdx_l = Repr.repr_l.demos_xdx[ref_demo_idx][0]
# end_xdx_l = Repr.repr_l.demos_xdx[ref_demo_idx][0]
# start_xdx_r = Repr.repr_r.demos_xdx[ref_demo_idx][0]
# end_xdx_r = Repr.repr_r.demos_xdx[ref_demo_idx][0]
# task_params = {'Left': {'start_xdx': start_xdx_l, 'end_xdx': end_xdx_l},
#                'Right': {'start_xdx': start_xdx_r, 'end_xdx': end_xdx_r}}
# traj_l, traj_r = Repr.generate(model_l, model_r, ref_demo_idx, task_params)
