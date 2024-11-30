"""
TP-GMM
=================

This example shows how to use the TP-GMM to learn a human demonstration motion.
"""
import numpy as np

import rofunc as rf

raw_demo = np.load('../data/LFD_ML/LeftHand.npy')
demos_x = [raw_demo[500:635, :], raw_demo[635:770, :], raw_demo[770:905, :]]

# --- TP-GMM ---
# Define the task parameters
start_xdx = [demos_x[i][0] for i in range(len(demos_x))]  # TODO: change to xdx
end_xdx = [demos_x[i][-1] for i in range(len(demos_x))]
task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
# Fit the model
Repr = rf.RofuncML.TPGMM(demos_x, task_params, plot=True)
model = Repr.fit()

# Reproductions for the same situations
traj, _ = Repr.reproduce(model, show_demo_idx=2)

# Reproductions for new situations: set the endpoint as the start point to make a cycled motion
ref_demo_idx = 2
start_xdx = [Repr.demos_xdx[ref_demo_idx][9]]
end_xdx = [Repr.demos_xdx[ref_demo_idx][9]]
Repr.task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
traj, _ = Repr.generate(model, ref_demo_idx)
