"""
CURI forward kinematics
========================

Forward kinematics and visualization of the CURI robot.
"""
import os

import numpy as np

import rofunc as rf
from rofunc.utils.oslab.path import get_rofunc_path

urdf_path = os.path.join(get_rofunc_path(), 'simulator/assets/urdf/curi/urdf/curi.urdf')
# urdf_path = os.path.join(get_rofunc_path(), 'simulator/assets/urdf/franka_description/robots/franka_panda.urdf')
link_name, joint_name, actuated_joint_name = rf.robolab.check_urdf(urdf_path)

joint_value = [0, 0, 0, 0, 0, 0, 0]
joint_value = np.array(joint_value)
robot, export_pose, cfg = rf.robolab.get_fk_from_model(urdf_path, actuated_joint_name, joint_value,
                                                       export_link='panda_left_link7')
print(export_pose)
