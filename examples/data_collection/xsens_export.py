"""
Xsens Export
================

This example shows how to export Xsens MVNX file to multiple skeleton .npy files.
"""
import rofunc as rf

# Export a single mvnx file to skeleton .npy files
mvnx_file = '../data/RAW_DEMO/xsens/010-003.mvnx'
rf.xsens.export(mvnx_file, output_type='segment')

# Export a batch of mvnx files in a directory
mvnx_dir = '../data/RAW_DEMO/xsens'
rf.xsens.export_batch(mvnx_dir)
