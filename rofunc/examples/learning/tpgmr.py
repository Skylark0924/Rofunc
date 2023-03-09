"""
TP-GMR for Tai Chi
=================

This example shows how to use the TP-GMR to learn a Tai Chi movement.
"""

import rofunc as rf
import numpy as np
import os

raw_demo = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LFD_ML/LeftHand.npy'))
raw_demo = np.expand_dims(raw_demo, axis=0)
demos_x = np.vstack((raw_demo[:, 82:232, :], raw_demo[:, 233:383, :], raw_demo[:, 376:526, :]))
show_demo_idx = 2
start_pose = demos_x[show_demo_idx][-1]
end_pose = demos_x[show_demo_idx][0]
model, rep = rf.tpgmr.uni(demos_x, show_demo_idx, start_pose, end_pose, plot=True)
