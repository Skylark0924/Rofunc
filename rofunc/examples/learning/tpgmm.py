"""
TP-GMM for Tai Chi
=================

This example shows how to use the TP-GMM to learn a Tai Chi movement.
"""

import rofunc as rf
import numpy as np
import os

left_raw_demo = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LFD_ML/LeftHand.npy'))
right_raw_demo = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LFD_ML/RightHand.npy'))
left_raw_demo = np.expand_dims(left_raw_demo, axis=0)
right_raw_demo = np.expand_dims(right_raw_demo, axis=0)
demos_left_x = np.vstack((left_raw_demo[:, 82:232, :], left_raw_demo[:, 233:383, :], left_raw_demo[:, 376:526, :]))
demos_right_x = np.vstack(
    (right_raw_demo[:, 82:232, :], right_raw_demo[:, 233:383, :], right_raw_demo[:, 376:526, :]))

model_l, model_r, rep_l, rep_r = rf.tpgmm.bi(demos_left_x, demos_right_x, show_demo_idx=2, plot=True)
