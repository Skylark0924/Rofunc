"""
CURI forward kinematics
========================

Forward kinematics and visualization of the CURI robot.
"""
import os

import numpy as np

import rofunc as rf
from rofunc.utils.file.path import get_rofunc_path

# urdf_path = os.path.join(get_rofunc_path(), 'simulator/assets/urdf/curi/urdf/curi.urdf')
urdf_path = os.path.join(get_rofunc_path(), 'simulator/assets/urdf/franka_description/robots/franka_panda.urdf')
link_name, joint_name, actuated_joint_name = rf.robolab.check_urdf(urdf_path)

# ['reference', 'summit_xls_front_right_wheel_joint', 'summit_xls_front_left_wheel_joint', 'summit_xls_back_left_wheel_joint',
# 'summit_xls_back_right_wheel_joint', 'torso_actuated_joint1', 'torso_actuated_joint2', 'torso_actuated_joint3',
# 'head_actuated_joint1', 'panda_left_joint1', 'panda_right_joint1', 'head_actuated_joint2', 'panda_left_joint2',
# 'panda_right_joint2', 'panda_right_joint3', 'panda_left_joint3', 'panda_left_joint4', 'panda_right_joint4',
# 'panda_left_joint5', 'panda_right_joint5', 'panda_right_joint6', 'panda_left_joint6', 'panda_left_joint7', 'panda_right_joint7',
# 'panda_left_finger_joint1', 'panda_right_finger_joint1']

# joint_value = [0.0, 1.0, 1.0, 1.0, 1.0, 2.1605735253736429e-16, 0.5326418108438093, -0.35713090036667317, 0.0,
#                -0.5248189935729404, 0.5248189935729395, 0.0, 1.1210417041164564, 1.1210417041164547,
#                1.9946571243973779, -1.9946571243973743, 0.729107826387619, 0.7291078263876194, 0.22015132006660315,
#                -0.2201513200666057, -0.1688895250088166, -0.1688895250088167, -0.14403010737637342,
#                0.14403010737637464, 0.0, 0.0]
joint_value = [0, 0, 0, 0, 0, 0, 0]
joint_value = np.array(joint_value)
robot, export_pose, cfg = rf.robolab.fk(urdf_path, actuated_joint_name, joint_value, export_link='panda_link7')
print(export_pose)

# robot.show(cfg=cfg)

# # Joint trajectory visualization
# joint_range = np.array([0.0, -np.pi / 4] * len(actuated_joint_name)).reshape((len(actuated_joint_name), 2))
# cfg_trajectory = {}
# for key, value in zip(actuated_joint_name, joint_range):
#     if key != 'reference':
#         cfg_trajectory[key] = value
# robot.animate(cfg_trajectory=cfg_trajectory)
