"""
Optitrack Export
================

This example shows how to clean the .csv files outputted by Optitrack.
"""

import rofunc as rf
import os
import numpy as np

input_path = os.path.join(rf.utils.get_rofunc_path(), 'data/RAW_DEMO/optitrack/Take 2023-03-29 06.23.40 PM.csv')
data = rf.optitrack.export(input_path)

root_dir = '/home/ubuntu/Data/optitrack'
# exp_name = 'rigid_body'
# np.save(os.path.join(root_dir, exp_name), data)
