"""
Visualize robots and objects
============================================================

This example shows how to visualize robots and objects in the Isaac Gym simulator in an interactive viewer.
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

pcd = o3d.io.read_point_cloud("/home/ubuntu/Github/Rofunc/rofunc/simulator/assets/urdf/ycb/011_banana/clouds/merged_cloud.ply")

o3d.visualization.draw_geometries([pcd],
                                  zoom=1,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])