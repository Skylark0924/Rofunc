"""
Coordinate transform
========================

This example shows how to transform coordinates between different reference frames and rotation representations.
"""

import rofunc as rf

q = rf.robolab.quaternion_multiply([1, 0, 0, 0], [-0.707, 0, 0, 0.707])
print(q)