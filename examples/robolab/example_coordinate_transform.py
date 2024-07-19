"""
Coordinate transform
========================

This example shows how to transform coordinates between different reference frames and rotation representations.
"""

import rofunc as rf

quat = [0.234, 0.23, 0.4, 1.3]

mat = rf.robolab.convert_ori_format(quat, "quat", "mat")
euler = rf.robolab.convert_ori_format(quat, "quat", "euler")
rf.logger.beauty_print(f"Quaternion: {rf.robolab.check_quat_tensor(quat)}")
rf.logger.beauty_print(f"Rotation matrix:\n {mat}")
rf.logger.beauty_print(f"Euler angles: {euler}")
# [Rofunc:INFO] Quaternion: tensor([[0.1672, 0.1644, 0.2859, 0.9291]])
# [Rofunc:INFO] Rotation matrix:
#  tensor([[[ 0.7825, -0.4763,  0.4011],
#          [ 0.5862,  0.7806, -0.2168],
#          [-0.2098,  0.4048,  0.8900]]])
# [Rofunc:INFO] Euler angles: tensor([[0.4268, 0.2114, 0.6430]])


rot = [[0.7825, -0.4763, 0.4011],
       [0.5862, 0.7806, -0.2168],
       [-0.2098, 0.4048, 0.8900]]
quat = rf.robolab.convert_ori_format(rot, "mat", "quat")
euler = rf.robolab.convert_ori_format(rot, "mat", "euler")
rf.logger.beauty_print(f"Rotation matrix:\n {rf.robolab.check_rot_matrix_tensor(rot)}")
rf.logger.beauty_print(f"Quaternion: {rf.robolab.check_quat_tensor(quat)}")
rf.logger.beauty_print(f"Euler angles: {euler}")
# [Rofunc:INFO] Rotation matrix:
#  tensor([[[ 0.7825, -0.4763,  0.4011],
#          [ 0.5862,  0.7806, -0.2168],
#          [-0.2098,  0.4048,  0.8900]]])
# [Rofunc:INFO] Quaternion: tensor([[0.1673, 0.1644, 0.2859, 0.9291]])
# [Rofunc:INFO] Euler angles: tensor([[0.4269, 0.2114, 0.6429]])


euler = [0.4268, 0.2114, 0.6430]
quat = rf.robolab.convert_ori_format(euler, "euler", "quat")
mat = rf.robolab.convert_ori_format(euler, "euler", "mat")
rf.logger.beauty_print(f"Euler angles: {rf.robolab.check_euler_tensor(euler)}")
rf.logger.beauty_print(f"Quaternion: {rf.robolab.check_quat_tensor(quat)}")
rf.logger.beauty_print(f"Rotation matrix:\n {mat}")
# [Rofunc:INFO] Euler angles: tensor([[0.4268, 0.2114, 0.6430]])
# [Rofunc:INFO] Quaternion: tensor([[0.1672, 0.1644, 0.2859, 0.9291]])
# [Rofunc:INFO] Rotation matrix:
#  tensor([[[ 0.7825, -0.4763,  0.4011],
#          [ 0.5863,  0.7806, -0.2168],
#          [-0.2098,  0.4047,  0.8900]]])
