"""
Xsens Visualize
================

This example shows how to use the skeleton files to visualize the human motion as a gif.
"""

import rofunc as rf

# Visualize a single skeleton file
skeleton_file = '/home/ubuntu/Data/xsens_mvnx/2023_05_16_joint_angle_test/010-003/segment'
rf.xsens.plot_skeleton(skeleton_file)
