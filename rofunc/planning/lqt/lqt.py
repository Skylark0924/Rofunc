from math import factorial

import numpy as np
from tqdm import tqdm
from typing import Union, Dict, Tuple


def get_matrices(param: Dict, data: np.ndarray):
    param['nbPoints'] = len(data)

    R = np.identity((param["nbData"] - 1) * param["nbVarPos"], dtype=np.float32) * param[
        "rfactor"]  # Control cost matrix

    tl = np.linspace(0, param["nbData"], param["nbPoints"] + 1)
    tl = np.rint(tl[1:]).astype(np.int64) - 1
    idx_slices = [slice(i, i + param["nb_var"], 1) for i in (tl * param["nb_var"])]

    # Target
    muQ = np.zeros((param["nb_var"] * param["nbData"], 1), dtype=np.float32)
    # Task precision
    Q = np.zeros((param["nb_var"] * param["nbData"], param["nb_var"] * param["nbData"]), dtype=np.float32)

    via_point = []
    for i in range(len(idx_slices)):
        slice_t = idx_slices[i]
        x_t = np.zeros((param["nb_var"], 1))
        x_t[:param["nbVarPos"]] = data[i].reshape((param["nbVarPos"], 1))
        muQ[slice_t] = x_t
        via_point.append(x_t)

        Q[slice_t, slice_t] = np.diag(
            np.hstack((np.ones(param["nbVarPos"]), np.zeros(param["nb_var"] - param["nbVarPos"]))))
    return via_point, muQ, Q, R, idx_slices, tl


def get_matrices_vel(param: Dict, data: np.ndarray):
    param['nbPoints'] = len(data)

    R = np.identity((param["nbData"] - 1) * param["nbVarPos"], dtype=np.float32) * param[
        "rfactor"]  # Control cost matrix

    tl = np.linspace(0, param["nbData"], param["nbPoints"] + 1)
    tl = np.rint(tl[1:]).astype(np.int64) - 1
    idx_slices = [slice(i, i + param["nb_var"], 1) for i in (tl * param["nb_var"])]

    # Target
    muQ = np.zeros((param["nb_var"] * param["nbData"], 1), dtype=np.float32)
    # Task precision
    Q = np.zeros((param["nb_var"] * param["nbData"], param["nb_var"] * param["nbData"]), dtype=np.float32)

    via_point = []
    for i in range(len(idx_slices)):
        slice_t = idx_slices[i]
        # x_t = np.zeros((param["nb_var"], 1))
        x_t = data[i].reshape((param["nb_var"], 1))
        muQ[slice_t] = x_t
        via_point.append(x_t)

        Q[slice_t, slice_t] = np.diag(
            np.hstack((np.ones(param["nbVarPos"]), np.zeros(param["nb_var"] - param["nbVarPos"]))))
    return via_point, muQ, Q, R, idx_slices, tl


def set_dynamical_system(param: Dict):
    A1d = np.zeros((param["nbDeriv"], param["nbDeriv"]), dtype=np.float32)
    B1d = np.zeros((param["nbDeriv"], 1), dtype=np.float32)
    for i in range(param["nbDeriv"]):
        A1d += np.diag(np.ones(param["nbDeriv"] - i), i) * param["dt"] ** i * 1 / factorial(i)
        B1d[param["nbDeriv"] - i - 1] = param["dt"] ** (i + 1) * 1 / factorial(i + 1)

    A = np.kron(A1d, np.identity(param["nbVarPos"], dtype=np.float32))
    B = np.kron(B1d, np.identity(param["nbVarPos"], dtype=np.float32))

    nb_var = param["nbVarPos"] * param["nbDeriv"]  # Dimension of state vector

    # Build Sx and Su transfer matrices
    Su = np.zeros((nb_var * param["nbData"], param["nbVarPos"] * (param["nbData"] - 1)))
    Sx = np.kron(np.ones((param["nbData"], 1)), np.eye(nb_var, nb_var))

    M = B
    for i in range(1, param["nbData"]):
        Sx[i * nb_var:param["nbData"] * nb_var, :] = np.dot(Sx[i * nb_var:param["nbData"] * nb_var, :], A)
        Su[nb_var * i:nb_var * i + M.shape[0], 0:M.shape[1]] = M
        M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]

    return Su, Sx


def get_u_x(param: Dict, start_pose: np.ndarray, muQ: np.ndarray, Q: np.ndarray, R: np.ndarray, Su: np.ndarray,
            Sx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x0 = start_pose.reshape((14, 1))

    # Equ. 18
    u_hat = np.linalg.inv(Su.T @ Q @ Su + R) @ Su.T @ Q @ (muQ - Sx @ x0)
    # x= S_x x_1 + S_u u
    x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1, param["nb_var"]))
    return u_hat, x_hat


def uni(param: Dict, data: np.ndarray):
    print('\033[1;32m--------{}--------\033[0m'.format('Planning smooth trajectory via LQT.'))

    start_pose = np.zeros((14,), dtype=np.float32)
    start_pose[:7] = data[0]

    via_point_pose = data[1:]
    param['nbPoints'] = len(via_point_pose)

    via_point, muQ, Q, R, idx_slices, tl = get_matrices(param, via_point_pose)
    Su, Sx = set_dynamical_system(param)
    u_hat, x_hat = get_u_x(param, start_pose, muQ, Q, R, Su, Sx)

    return u_hat, x_hat, muQ, idx_slices


def uni_hierarchical(param: Dict, data: np.ndarray, interval: int = 3):
    print('\033[1;32m--------{}--------\033[0m'.format('Planning smooth trajectory via LQT hierarchically.'))

    start_pose = np.zeros((14,), dtype=np.float32)
    start_pose[:7] = data[0, :7]

    x_hat_lst = []
    for i in tqdm(range(0, len(data), interval)):
        via_point_pose = data[i + 1:i + interval + 1]
        param['nbPoints'] = len(via_point_pose)

        via_point, muQ, Q, R, idx_slices, tl = get_matrices_vel(param, via_point_pose)
        Su, Sx = set_dynamical_system(param)
        u_hat, x_hat = get_u_x(param, start_pose, muQ, Q, R, Su, Sx)
        start_pose = x_hat[-1]
        x_hat_lst.append(x_hat)

    x_hat = np.array(x_hat_lst).reshape((-1, 14))
    return u_hat, x_hat, muQ, idx_slices


def bi(param, l_data, r_data):
    print('\033[1;32m--------{}--------\033[0m'.format('Planning smooth bimanual trajectory via LQT.'))

    l_start_pose = np.zeros((14,), dtype=np.float32)
    r_start_pose = np.zeros((14,), dtype=np.float32)
    l_start_pose[:7] = l_data[0]
    r_start_pose[:7] = r_data[0]
    via_point_pose_l = l_data[1:]
    via_point_pose_r = r_data[1:]
    param['nbPoints'] = len(via_point_pose_l)

    via_point_l, muQ_l, Q, R, idx_slices, tl = get_matrices(param, via_point_pose_l)
    via_point_r, muQ_r, Q, R, idx_slices, tl = get_matrices(param, via_point_pose_r)

    Su, Sx = set_dynamical_system(param)

    u_hat_l, x_hat_l = get_u_x(param, l_start_pose, muQ_l, Q, R, Su, Sx)
    u_hat_r, x_hat_r = get_u_x(param, r_start_pose, muQ_r, Q, R, Su, Sx)
    return u_hat_l, u_hat_r, x_hat_l, x_hat_r, muQ_l, muQ_r, idx_slices


if __name__ == '__main__':
    import rofunc as rf

    param = {
        "nbData": 200,  # Number of data points
        "nbVarPos": 7,  # Dimension of position data
        "nbDeriv": 2,  # Number of static and dynamic features (2 -> [x,dx])
        "dt": 1e-2,  # Time step duration
        "rfactor": 1e-8  # Control cost
    }
    param["nb_var"] = param["nbVarPos"] * param["nbDeriv"]  # Dimension of state vector

    # Uni
    # data = np.load(
    #     '/home/ubuntu/Github/DGform/interactive/skylark/stretch-31-Aug-2022-08:48:15.683806/z_manipulator_poses.npy')
    # filter_indices = [0, 1, 5, 10, 22, 36]
    # data = data[filter_indices]

    # u_hat, x_hat, muQ, idx_slices = rf.lqt.uni(param, data)
    # rf.lqt.plot_3d_uni(x_hat, muQ, idx_slices, ori=False, save=False)

    # Bi
    # data = np.loadtxt('/home/ubuntu/Github/DGform/controller/data//link7_loc_ori.txt', delimiter=', ')
    # l_data = data[0:len(data):2]
    # r_data = data[1:len(data):2]
    # u_hat_l, u_hat_r, x_hat_l, x_hat_r, muQ_l, muQ_r, idx_slices = rf.lqt.bi(param, l_data, r_data)
    # rf.lqt.plot_3d_bi(x_hat_l, x_hat_r, muQ_l, muQ_r, idx_slices, ori=False, save=False)

    # Recursive
    data_raw = np.load('/home/ubuntu/Data/2022_09_09_Taichi/rep3_r.npy')
    data = np.zeros((len(data_raw), 14))
    data[:, :7] = data_raw
    filter_indices = [i for i in range(0, len(data_raw) - 10, 5)]
    filter_indices.append(len(data_raw) - 1)
    data = data[filter_indices]
    u_hat, x_hat, muQ, idx_slices = uni_hierarchical(param, data, interval=2)
    rf.lqt.plot_3d_uni(x_hat, ori=False, save=True, save_file_name='/home/ubuntu/Data/2022_09_09_Taichi/lqt_rep3_r.npy')

    # Show the data
    # data = np.load('/home/ubuntu/Data/2022_09_09_Taichi/lqt_rep3_l.npy')
    # rf.lqt.plot_3d_uni(data)
