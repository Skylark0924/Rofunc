"""
Xsens Export
================

This example shows how to export Xsens MVNX file to multiple skeleton .npy files.
"""

import rofunc as rf

# Export a single mvnx file
mvnx_file = '../xsens_data/test.mvnx'
rf.xsens.export(mvnx_file)

# Export a batch of mvnx files in a directory
mvnx_dir = '../xsens_data'
rf.xsens.export_batch(mvnx_dir)
