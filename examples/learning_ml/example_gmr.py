"""
GMR
=================

This example shows how to use the GMR to learn a human demonstration motion.
"""
import numpy as np

import rofunc as rf

data = np.load('../../data/LFD_ML/pbd/' + 'test_001.npy', allow_pickle=True, encoding="latin1")[()]

demos_x = data['x']  # Position data
demos_dx = data['dx']  # Velocity data
demos_xdx = [np.hstack([_x, _dx]) for _x, _dx in zip(demos_x, demos_dx)]  # Position-velocity

representation = rf.RofuncML.GMR(demos_x, demos_dx, demos_xdx, plot=True)
representation.fit()
