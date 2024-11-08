import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def quaternion_to_rotation_matrix(quaternion):
    return R.from_quat(quaternion).as_matrix()

def forward_kinematics(joint_positions, joint_rotations, parent_indices):
    num_joints = len(joint_positions)
    global_positions = np.zeros_like(joint_positions)
    global_rotations = [np.eye(3) for _ in range(num_joints)]
    for i in range(num_joints):
        if parent_indices[i] == -1:
            global_positions[i] = joint_positions[i]
            global_rotations[i] = quaternion_to_rotation_matrix(joint_rotations[i])
        else:
            parent_index = parent_indices[i]
            parent_rotation = global_rotations[parent_index]
            parent_position = global_positions[parent_index]
            rotation_matrix = quaternion_to_rotation_matrix(joint_rotations[i])
            global_rotations[i] = parent_rotation @ rotation_matrix
            global_positions[i] = parent_position + parent_rotation @ joint_positions[i]

    # Adjust all joints so the feet are on the ground
    left_foot_index = 14  # Replace with the actual index of LeftFoot
    right_foot_index = 11  # Replace with the actual index of RightFoot
    min_z = min(global_positions[left_foot_index, 2], global_positions[right_foot_index, 2])
    global_positions[:, 2] -= min_z

    return global_positions, global_rotations

# def forward_kinematics(joint_positions, joint_rotations, parent_indices):
#     num_joints = len(joint_positions)
#     global_positions = np.zeros_like(joint_positions)
#     global_rotations = [np.eye(3) for _ in range(num_joints)]
#     for i in range(num_joints):
#         if parent_indices[i] == -1:
#             global_positions[i] = joint_positions[i]
#             global_rotations[i] = quaternion_to_rotation_matrix(joint_rotations[i])
#         else:
#             parent_index = parent_indices[i]
#             parent_rotation = global_rotations[parent_index]
#             parent_position = global_positions[parent_index]
#             rotation_matrix = quaternion_to_rotation_matrix(joint_rotations[i])
#             global_rotations[i] = parent_rotation @ rotation_matrix
#             global_positions[i] = parent_position + parent_rotation @ joint_positions[i]
#     return global_positions, global_rotations

def calculate_manipulability(jacobian):
    M_v = jacobian @ jacobian.T
    M_F = np.linalg.inv(M_v[3:, 3:])
    eigenvalues, eigenvectors = np.linalg.eigh(M_F)
    return eigenvalues, eigenvectors