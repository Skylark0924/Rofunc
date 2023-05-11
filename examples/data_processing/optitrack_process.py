"""
Optitrack Export
================

This example shows how to get the optitrack data from the csv file.
"""
import rofunc as rf
import os.path as osp

input_file = 'examples/data_collection/optitrack.csv'
parent_dir = osp.dirname(input_file)
objs, meta = rf.optitrack.get_objects(input_file)
# objs, meta are lists. If input_file points to a folder, each element of objs and meta is the data corresponding to one file.
# In a folder, only the file with the following name format are considered: 'Take*.csv'
# If input_file points to a file, objs and meta are lists with only one element.

objs = objs[0]
meta = meta[0]

# Visualize objects
# The pyplot animation produced can be paused and resumed by pressing any keyboard key.
rf.optitrack.visualize_objects(parent_dir, objs, meta)

#### Option 1: Pandas DataFrame
#### ---------------------------

data = rf.optitrack.get_time_series(input_path, meta[0])

larm_pos_x = data.iloc[:, objs[0]['left_arm']['pose']['Position']['X']]


#### Option 2: numpy array
#### ---------------------------
del_objects = ['cup', 'hand_right']

# You can delete object from the dictionnary and they will not be considered in the data
for obj in del_objects:
    del objs[obj]

parent_dir = osp.dirname(input_file)
data, labels = rf.optitrack.data_clean(parent_dir, legacy=False, objects=objs)

# data is a numpy array of shape (n_samples, n_features)
# labels is a list of strings corresponding to the name of the features
#
# Accessing the position and attitude of an object over all samples:
# Coordinates names and order: ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

data_ptr = labels.index('left_arm.pose.x')
assert data_ptr + 6 == labels.index('left_arm.pose.qw')
larm_pose = data[:, data_ptr:data_ptr+7]
