from scipy.spatial.transform import Rotation
import numpy as np

# <editor-fold desc="CURI Matrices Definition">
# Matrix from world frame to left arm base frame for CURI robot
T_W2CURIl = np.array([[0.626009, -0.736984, 0.254887, 0.45908], [-0.12941, 0.224144, 0.965926, 0.11216],
                      [-0.769003, -0.637663, 0.0449435, 1.19266], [0, 0, 0, 1]])
# Matrix from world frame to right arm base frame
T_W2CURIr = np.array([[0.626009, 0.736984, 0.254887, 0.45908], [0.12941, 0.224144, -0.965926, -0.11216],
                      [-0.769003, 0.637663, 0.0449435, 1.19266], [0, 0, 0, 1]])

# Matrix from table frame to world frame for CURI robot
T_table2W = np.array([[0, -1, 0, 0.5], [0, 0, 1, -0.65],
                      [-1, 0, 0, 1.25], [0, 0, 0, 1]])

# Matrix to transform ori of robot ee for CURI robot
T_rotate_l = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
T_rotate_r = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])

# Inverse Matrices
T_W2CURIl_inv = np.linalg.inv(T_W2CURIl)
T_W2CURIr_inv = np.linalg.inv(T_W2CURIr)
T_table2W_inv = np.linalg.inv(T_table2W)


# print(T_l_inv)
# </editor-fold>


def scale(data, initial_robot_pose, raw_traj):
    data_ori = data[:, 3:7]
    data_pos = initial_robot_pose + 0.4 * (data[0:3] - raw_traj[0:3])
    data_pos[1] = data_pos[1] + 1.6 * (data[1] - raw_traj[1])
    data_ori_new = np.append(data_ori[1:4], data_ori[0])
    data_ori = np.append(data_ori_new[0], data_ori_new[3])
    data_ori = np.append(data_ori, data_ori_new[2])
    data_ori = np.append(data_ori, data_ori_new[1])
    return data_pos, data_ori


def transTable2Arm(data, T_table2W_inv, T_W2arm_inv):
    """
    Transform pose w.r.t table frame to arm base frame
    Args:
        data: trajectory data
        T_table2W_inv: inverse transformation matrix from table to world frame
        T_W2arm_inv: inverse transformation matrix from world to arm base frame
    Returns:
        transformed trajectory w.r.t arm base frame
    """
    data_pos = data[:, 0:3]
    data_ori = data[:, 3:7]
    data_pos_transform = np.dot(T_table2W_inv[0:3, 0:3], data_pos) + (T_table2W_inv[0:3, 3:4]).T
    data_pos_transform = (np.dot(T_W2arm_inv[0:3, 0:3], data_pos_transform.T)).T + (T_W2arm_inv[0:3, 3:4]).T
    data_pos_transform = np.reshape(data_pos_transform, -1)
    data_ori_transform = np.dot(T_table2W_inv[0:3, 0:3], Rotation.from_quat(data_ori).as_matrix())
    data_ori_transform = np.dot(T_W2arm_inv[0:3, 0:3], data_ori_transform)
    data_ori_transform = Rotation.from_matrix(data_ori_transform).as_quat()
    return data_pos_transform, data_ori_transform


def transTable2World(data, T_table2W_inv):
    """
    Transform pose w.r.t table frame to world frame
    Args:
        data: trajectory data
        T_table2W_inv: inverse transformation matrix from table to world frame
    Returns:
        transformed trajectory w.r.t world frame
    """
    data_pos = data[:, 0:3].T
    data_ori = data[:, 3:7].T
    data_pos_transform = np.dot(T_table2W_inv[0:3, 0:3], data_pos) + (T_table2W_inv[0:3, 3:4]).T
    data_pos_transform = np.reshape(data_pos_transform, -1)
    data_ori_transform = np.dot(T_table2W_inv[0:3, 0:3], Rotation.from_quat(data_ori).as_matrix())
    data_ori_transform = Rotation.from_matrix(data_ori_transform).as_quat()
    return data_pos_transform, data_ori_transform
