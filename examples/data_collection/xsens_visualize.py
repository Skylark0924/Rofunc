"""
Xsens Visualize
================

This example shows how to use the skeleton files to visualize the human motion as a gif.
"""
import rofunc as rf

# Visualize a single skeleton directory
skeleton_path = '../data/RAW_DEMO/xsens/010-004/segment'
rf.xsens.plot_skeleton(skeleton_path, save_gif=True)

# Visualize a batch of skeleton directories
skeleton_dir = '../data/RAW_DEMO/xsens'
rf.xsens.plot_skeleton_batch(skeleton_dir, save_gif=False)
