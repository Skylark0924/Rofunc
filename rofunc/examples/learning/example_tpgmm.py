"""
TP-GMM
=================

This example shows how to use the TP-GMM to learn a human demonstration motion.
"""
import os
import numpy as np
import rofunc as rf

raw_demo = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LFD_ML/LeftHand.npy'))
demos_x = [raw_demo[500:635, :], raw_demo[635:770, :], raw_demo[770:905, :]]

# TP-GMM
representation = rf.lfd.tpgmm.TPGMM(demos_x, plot=True)
model = representation.fit()

# Reproductions for the same situations
traj = representation.reproduce(model, show_demo_idx=2)

# Reproductions for new situations: set the endpoint as the start point to make a cycled motion
ref_demo_idx = 2
start_xdx = representation.demos_xdx[ref_demo_idx][0]
end_xdx = representation.demos_xdx[ref_demo_idx][0]
task_params = {'start_xdx': start_xdx, 'end_xdx': end_xdx}
traj = representation.generate(model, ref_demo_idx, task_params)
