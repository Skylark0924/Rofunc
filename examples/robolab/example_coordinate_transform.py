"""
Coordinate transform
========================

This example shows how to transform coordinates between different reference frames and rotation representations.
"""

import rofunc as rf
import numpy as np

Rot_quat = rf.robolab.quaternion_about_axis(-np.pi / 2, [0, 1, 0])
Rot_matrix = rf.robolab.homo_matrix_from_quaternion(Rot_quat)

q = rf.robolab.quaternion_multiply([1, 0, 0, 0], Rot_quat)
print(q)
