"""
Optitrack Export
================

This example shows how to clean the .csv files outputted by Optitrack.
"""

import rofunc as rf
import os
import numpy as np

data = rf.optitrack.export('/home/ubuntu/Data/optitrack/Take 2022-12-21 05.41.07 PM.csv')

root_dir = '/home/ubuntu/Data/optitrack'
exp_name = 'rigid_body'
np.save(os.path.join(root_dir, exp_name), data)

# print(data)

data = rf.optitrack.export('/home/ubuntu/Data/optitrack/Take 2022-12-21 05.41.07 PM.csv')
print(data)
