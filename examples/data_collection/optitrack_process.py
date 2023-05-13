"""
Optitrack Export
================

This example shows how to get the optitrack data from the csv file.
"""
import rofunc as rf
import os

# If input_path points to a folder, each element of objs and meta is the data corresponding to one file.
# In a folder, only the file with the following name format are considered: 'Take*.csv'
# If input_file points to a file, objs and meta are lists with only one element.
input_path = os.path.join(rf.utils.get_rofunc_path(), 'data/RAW_DEMO/optitrack/Take 2023-03-29 06.23.40 PM.csv')
parent_dir = os.path.dirname(input_path)
objs_list, meta_list = rf.optitrack.get_objects(input_path)

# Visualize objects
# The pyplot animation produced can be paused and resumed by pressing any keyboard key.
objs = objs_list[0]
meta = meta_list[0]
rf.optitrack.visualize_objects(parent_dir, objs, meta, up_axis='Y')

# --- Option 1: Pandas DataFrame ---
data = rf.optitrack.get_time_series(parent_dir, meta_list[0])
box_pos_x = data.iloc[:, objs_list[0]['box']['pose']['Position']['X']]

# --- Option 2: numpy array ---
del_objects = ['left']

# You can delete object from the dictionary, and they will not be considered in the data
for obj in del_objects:
    del objs[obj]

# data is a numpy array of shape (n_samples, n_features)
# labels is a list of strings corresponding to the name of the features

data, labels = rf.optitrack.data_clean(parent_dir, legacy=False, objs=objs_list[0])[0]

# Accessing the position and attitude of an object over all samples:
# Coordinates names and order: ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
data_ptr = labels.index('box.pose.x')
assert data_ptr + 6 == labels.index('box.pose.qw')
box_pos_x = data[:, data_ptr:data_ptr + 7]
