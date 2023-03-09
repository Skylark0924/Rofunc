"""
TP-GMM
=================

This example shows how to use the TP-GMM to learn a human demonstration motion.
"""
import os
import numpy as np
import rofunc as rf

raw_demo = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LFD_ML/LeftHand.npy'))
raw_demo = np.expand_dims(raw_demo, axis=0)
demos_x = np.vstack((raw_demo[:, 82:232, :], raw_demo[:, 233:383, :], raw_demo[:, 376:526, :]))

representation = rf.lfd.tpgmm.TPGMM(demos_x)
model = representation.fit(plot=True)

# Reproductions for the same situations
traj = representation.reproduce(model, show_demo_idx=2, plot=True)

# Reproductions for new situations
ref_demo_idx = 2
A, b = representation.demos_A_xdx[ref_demo_idx][0], representation.demos_b_xdx[ref_demo_idx][0]
b[1] = b[0]
task_params = {'A': A, 'b': b}
traj = representation.generate(model, ref_demo_idx, task_params, plot=True)
