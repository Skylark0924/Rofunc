"""
Optitrack Visualize
================

This example shows how to visualize Optitrack data.
"""
import os
import rofunc as rf


input_path = '../data/RAW_DEMO/optitrack/Take 2023-03-29 06.23.40 PM.csv'
parent_dir = os.path.dirname(input_path)
objs_list, meta_list = rf.optitrack.get_objects(input_path)

# --- Visualize objects ---
# The pyplot animation produced can be paused and resumed by pressing any keyboard key.
rf.optitrack.visualize_objects(parent_dir, objs_list[0], meta_list[0], up_axis='Y')
