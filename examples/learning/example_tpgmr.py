"""
TP-GMR for Tai Chi
=================

This example shows how to use the TP-GMR to learn a Tai Chi movement.
"""
import os
import numpy as np
import rofunc as rf

raw_demo = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LFD_ML/LeftHand.npy'))
demos_x = [raw_demo[500:635, :], raw_demo[635:770, :], raw_demo[770:905, :]]

# TP-GMR
representation = rf.learning.tpgmr.TPGMR(demos_x, plot=True)
model = representation.fit()

# Reproductions for the same situations
traj, _ = representation.reproduce(model, show_demo_idx=2)

# Reproductions for new situations
ref_demo_idx = 2
start_xdx = representation.demos_xdx[ref_demo_idx][0]
end_xdx = representation.demos_xdx[ref_demo_idx][0]
task_params = {'start_xdx': start_xdx, 'end_xdx': end_xdx}
traj, _ = representation.generate(model, ref_demo_idx, task_params)


# raw_demo = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LFD_ML/LeftHand.npy'))
# demos_left_x = [raw_demo[500:635, :], raw_demo[635:770, :], raw_demo[770:905, :]]
# raw_demo = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LFD_ML/RightHand.npy'))
# demos_right_x = [raw_demo[500:635, :], raw_demo[635:770, :], raw_demo[770:905, :]]
#
# representation = rf.learning.tpgmr.TPGMRBi(demos_left_x, demos_right_x, horizon=300, plot=True)
# model_l, model_r = representation.fit()
#
# # Reproductions for the same situations
# traj_l, traj_r = representation.reproduce(model_l, model_r, show_demo_idx=2)
#
# # Reproductions for new situations
# ref_demo_idx = 2
# start_xdx_l = representation.repr_l.demos_xdx[ref_demo_idx][0]
# end_xdx_l = representation.repr_l.demos_xdx[ref_demo_idx][0]
# start_xdx_r = representation.repr_r.demos_xdx[ref_demo_idx][0]
# end_xdx_r = representation.repr_r.demos_xdx[ref_demo_idx][0]
# task_params = {'Left': {'start_xdx': start_xdx_l, 'end_xdx': end_xdx_l},
#                'Right': {'start_xdx': start_xdx_r, 'end_xdx': end_xdx_r}}
# traj_l, traj_r = representation.generate(model_l, model_r, ref_demo_idx, task_params)
