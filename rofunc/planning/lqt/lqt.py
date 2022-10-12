"""
    Linear Quadratic tracker
"""
from math import factorial
from typing import Tuple

import numpy as np
from rofunc.config.get_config import *
from tqdm import tqdm


def get_matrices(cfg: DictConfig, data: np.ndarray):
    cfg.nbPoints = len(data)

    R = np.identity((cfg.nbData - 1) * cfg.nbVarPos, dtype=np.float32) * cfg.rfactor  # Control cost matrix

    tl = np.linspace(0, cfg.nbData, cfg.nbPoints + 1)
    tl = np.rint(tl[1:]).astype(np.int64) - 1
    idx_slices = [slice(i, i + cfg.nbVar, 1) for i in (tl * cfg.nbVar)]

    # Target
    muQ = np.zeros((cfg.nbVar * cfg.nbData, 1), dtype=np.float32)
    # Task precision
    Q = np.zeros((cfg.nbVar * cfg.nbData, cfg.nbVar * cfg.nbData), dtype=np.float32)

    via_point = []
    for i in range(len(idx_slices)):
        slice_t = idx_slices[i]
        x_t = np.zeros((cfg.nbVar, 1))
        x_t[:cfg.nbVarPos] = data[i].reshape((cfg.nbVarPos, 1))
        muQ[slice_t] = x_t
        via_point.append(x_t)

        Q[slice_t, slice_t] = np.diag(
            np.hstack((np.ones(cfg.nbVarPos), np.zeros(cfg.nbVar - cfg.nbVarPos))))
    return via_point, muQ, Q, R, idx_slices, tl


def get_matrices_vel(cfg: DictConfig, data: np.ndarray):
    cfg.nbPoints = len(data)

    R = np.identity((cfg.nbData - 1) * cfg.nbVarPos, dtype=np.float32) * cfg.rfactor  # Control cost matrix

    tl = np.linspace(0, cfg.nbData, cfg.nbPoints + 1)
    tl = np.rint(tl[1:]).astype(np.int64) - 1
    idx_slices = [slice(i, i + cfg.nbVar, 1) for i in (tl * cfg.nbVar)]

    # Target
    muQ = np.zeros((cfg.nbVar * cfg.nbData, 1), dtype=np.float32)
    # Task precision
    Q = np.zeros((cfg.nbVar * cfg.nbData, cfg.nbVar * cfg.nbData), dtype=np.float32)

    via_point = []
    for i in range(len(idx_slices)):
        slice_t = idx_slices[i]
        # x_t = np.zeros((cfg.nbVar, 1))
        x_t = data[i].reshape((cfg.nbVar, 1))
        muQ[slice_t] = x_t
        via_point.append(x_t)

        Q[slice_t, slice_t] = np.diag(
            np.hstack((np.ones(cfg.nbVarPos), np.zeros(cfg.nbVar - cfg.nbVarPos))))
    return via_point, muQ, Q, R, idx_slices, tl


def set_dynamical_system(cfg: DictConfig):
    A1d = np.zeros((cfg.nbDeriv, cfg.nbDeriv), dtype=np.float32)
    B1d = np.zeros((cfg.nbDeriv, 1), dtype=np.float32)
    for i in range(cfg.nbDeriv):
        A1d += np.diag(np.ones(cfg.nbDeriv - i), i) * cfg.dt ** i * 1 / factorial(i)
        B1d[cfg.nbDeriv - i - 1] = cfg.dt ** (i + 1) * 1 / factorial(i + 1)

    A = np.kron(A1d, np.identity(cfg.nbVarPos, dtype=np.float32))
    B = np.kron(B1d, np.identity(cfg.nbVarPos, dtype=np.float32))

    nb_var = cfg.nbVar  # Dimension of state vector

    # Build Sx and Su transfer matrices
    Su = np.zeros((nb_var * cfg.nbData, cfg.nbVarPos * (cfg.nbData - 1)))
    Sx = np.kron(np.ones((cfg.nbData, 1)), np.eye(nb_var, nb_var))

    M = B
    for i in range(1, cfg.nbData):
        Sx[i * nb_var:cfg.nbData * nb_var, :] = np.dot(Sx[i * nb_var:cfg.nbData * nb_var, :], A)
        Su[nb_var * i:nb_var * i + M.shape[0], 0:M.shape[1]] = M
        M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]

    return Su, Sx


def get_u_x(cfg: DictConfig, start_pose: np.ndarray, muQ: np.ndarray, Q: np.ndarray, R: np.ndarray, Su: np.ndarray,
            Sx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x0 = start_pose.reshape((cfg.nbVar, 1))

    # Equ. 18
    u_hat = np.linalg.inv(Su.T @ Q @ Su + R) @ Su.T @ Q @ (muQ - Sx @ x0)
    # x= S_x x_1 + S_u u
    x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1, cfg.nbVar))
    return u_hat, x_hat


def uni(data: np.ndarray, cfg: DictConfig = None):
    print('\033[1;32m--------{}--------\033[0m'.format('Planning smooth trajectory via LQT'))

    cfg = get_config("./", "lqt") if cfg is None else cfg

    start_pose = np.zeros((cfg.nbVar,), dtype=np.float32)
    start_pose[:cfg.nbVarPos] = data[0]

    via_point_pose = data[1:]
    cfg.nbPoints = len(via_point_pose)

    via_point, muQ, Q, R, idx_slices, tl = get_matrices(cfg, via_point_pose)
    Su, Sx = set_dynamical_system(cfg)
    u_hat, x_hat = get_u_x(cfg, start_pose, muQ, Q, R, Su, Sx)

    return u_hat, x_hat, muQ, idx_slices


def uni_hierarchical(data: np.ndarray, cfg: DictConfig = None, interval: int = 3):
    print('\033[1;32m--------{}--------\033[0m'.format('Planning smooth trajectory via LQT hierarchically'))

    cfg = get_config("./", "lqt") if cfg is None else cfg

    start_pose = np.zeros((cfg.nbVar,), dtype=np.float32)
    start_pose[:cfg.nbVarPos] = data[0, :cfg.nbVarPos]

    x_hat_lst = []
    for i in tqdm(range(0, len(data), interval)):
        via_point_pose = data[i + 1:i + interval + 1]
        cfg.nbPoints = len(via_point_pose)

        via_point, muQ, Q, R, idx_slices, tl = get_matrices_vel(cfg, via_point_pose)
        Su, Sx = set_dynamical_system(cfg)
        u_hat, x_hat = get_u_x(cfg, start_pose, muQ, Q, R, Su, Sx)
        start_pose = x_hat[-1]
        x_hat_lst.append(x_hat)

    x_hat = np.array(x_hat_lst).reshape((-1, cfg.nbVarX * cfg.nbDeriv))
    return u_hat, x_hat, muQ, idx_slices


def bi(l_data: np.ndarray, r_data: np.ndarray, cfg: DictConfig = None):
    print('\033[1;32m--------{}--------\033[0m'.format('Planning smooth bimanual trajectory via LQT'))

    cfg = get_config("./", "lqt") if cfg is None else cfg

    l_start_pose = np.zeros((cfg.nbVar,), dtype=np.float32)
    r_start_pose = np.zeros((cfg.nbVar,), dtype=np.float32)
    l_start_pose[:cfg.nbVarPos] = l_data[0]
    r_start_pose[:cfg.nbVarPos] = r_data[0]
    via_point_pose_l = l_data[1:]
    via_point_pose_r = r_data[1:]
    cfg.nbPoints = len(via_point_pose_l)

    via_point_l, muQ_l, Q, R, idx_slices, tl = get_matrices(cfg, via_point_pose_l)
    via_point_r, muQ_r, Q, R, idx_slices, tl = get_matrices(cfg, via_point_pose_r)

    Su, Sx = set_dynamical_system(cfg)

    u_hat_l, x_hat_l = get_u_x(cfg, l_start_pose, muQ_l, Q, R, Su, Sx)
    u_hat_r, x_hat_r = get_u_x(cfg, r_start_pose, muQ_r, Q, R, Su, Sx)
    return u_hat_l, u_hat_r, x_hat_l, x_hat_r, muQ_l, muQ_r, idx_slices


if __name__ == '__main__':
    import rofunc as rf

    # with initialize(config_path="../../config", version_base=None):
    #     cfg = compose(config_name="lqt")

    # <editor-fold desc="Uni example">
    data = np.load(
        '/home/ubuntu/Github/DGform/interactive/skylark/stretch-31-Aug-2022-08:48:15.683806/z_manipulator_poses.npy')
    filter_indices = [0, 1, 5, 10, 22, 36]
    data = data[filter_indices]
    u_hat, x_hat, muQ, idx_slices = rf.lqt.uni(data)
    rf.lqt.plot_3d_uni(x_hat, muQ, idx_slices, ori=False, save=False)
    # </editor-fold>

    # <editor-fold desc="Bi example">
    # data = np.loadtxt('/home/ubuntu/Github/DGform/controller/data//link7_loc_ori.txt', delimiter=', ')
    # l_data = data[0:len(data):2]
    # r_data = data[1:len(data):2]
    # u_hat_l, u_hat_r, x_hat_l, x_hat_r, muQ_l, muQ_r, idx_slices = bi(l_data, r_data)
    # rf.lqt.plot_3d_bi(x_hat_l, x_hat_r, muQ_l, muQ_r, idx_slices, ori=False, save=False)
    # </editor-fold>

    # <editor-fold desc="Recursive example">
    # data_raw = np.load('/home/ubuntu/Data/2022_09_09_Taichi/rep3_r.npy')
    # data = np.zeros((len(data_raw), 14))
    # data[:, :7] = data_raw
    # filter_indices = [i for i in range(0, len(data_raw) - 10, 5)]
    # filter_indices.append(len(data_raw) - 1)
    # data = data[filter_indices]
    # u_hat, x_hat, muQ, idx_slices = uni_hierarchical(data, interval=2)
    # rf.lqt.plot_3d_uni(x_hat, ori=False, save=True, save_file_name='/home/ubuntu/Data/2022_09_09_Taichi/lqt_rep3_r.npy')
    # </editor-fold>
