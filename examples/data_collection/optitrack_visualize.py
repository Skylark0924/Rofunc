"""
Optitrack Visualize
================

This example shows how to visualize Optitrack data.
"""

import rofunc as rf

input_path = '/home/ubuntu/Data/optitrack_record/2023_03_14'
objs, meta = rf.optitrack.get_objects(input_path)

rf.optitrack.plot_objects(input_path, objs[0], meta[0])
