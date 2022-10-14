"""
    Linear Quadratic tracker

    Refers to https://gitlab.idiap.ch/rli/robotics-codes-from-scratch by Dr. Sylvain Calinon
"""
from typing import Tuple

import numpy as np
from math import factorial
from tqdm import tqdm

from rofunc.config.get_config import *


def get_matrices(cfg: DictConfig, via_points: np.ndarray):
    cfg.nbPoints = len(via_points)

    # Control cost matrix
    R = np.identity((cfg.nbData - 1) * cfg.nbVarPos, dtype=np.float32) * cfg.rfactor

    tl = np.linspace(0, cfg.nbData, cfg.nbPoints + 1)
    tl = np.rint(tl[1:]).astype(np.int64) - 1
    idx_slices = [slice(i, i + cfg.nbVar, 1) for i in (tl * cfg.nbVar)]

    # Target
    mu = np.zeros((cfg.nbVar * cfg.nbData, 1), dtype=np.float32)
    # Task precision
    Q = np.zeros((cfg.nbVar * cfg.nbData, cfg.nbVar * cfg.nbData), dtype=np.float32)

    for i in range(len(idx_slices)):
        slice_t = idx_slices[i]
        x_t = via_points[i].reshape((cfg.nbVar, 1))
        mu[slice_t] = x_t

        Q[slice_t, slice_t] = np.diag(np.hstack((np.ones(cfg.nbVarPos), np.zeros(cfg.nbVar - cfg.nbVarPos))))
    return mu, Q, R, idx_slices, tl


def set_dynamical_system(cfg: DictConfig):
    A1d = np.zeros((cfg.nbDeriv, cfg.nbDeriv), dtype=np.float32)
    B1d = np.zeros((cfg.nbDeriv, 1), dtype=np.float32)
    for i in range(cfg.nbDeriv):
        A1d += np.diag(np.ones(cfg.nbDeriv - i), i) * cfg.dt ** i * 1 / factorial(i)
        B1d[cfg.nbDeriv - i - 1] = cfg.dt ** (i + 1) * 1 / factorial(i + 1)

    A = np.kron(A1d, np.identity(cfg.nbVarPos, dtype=np.float32))
    B = np.kron(B1d, np.identity(cfg.nbVarPos, dtype=np.float32))

    # Build Sx and Su transfer matrices
    Su = np.zeros((cfg.nbVar * cfg.nbData, cfg.nbVarPos * (cfg.nbData - 1)))
    Sx = np.kron(np.ones((cfg.nbData, 1)), np.eye(cfg.nbVar, cfg.nbVar))

    M = B
    for i in range(1, cfg.nbData):
        Sx[i * cfg.nbVar:cfg.nbData * cfg.nbVar, :] = np.dot(Sx[i * cfg.nbVar:cfg.nbData * cfg.nbVar, :], A)
        Su[cfg.nbVar * i:cfg.nbVar * i + M.shape[0], 0:M.shape[1]] = M
        M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]
    return Su, Sx


def get_u_x(cfg: DictConfig, start_pose: np.ndarray, mu: np.ndarray, Q: np.ndarray, R: np.ndarray, Su: np.ndarray,
            Sx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x0 = start_pose.reshape((cfg.nbVar, 1))
    # Equ. 18
    u_hat = np.linalg.inv(Su.T @ Q @ Su + R) @ Su.T @ Q @ (mu - Sx @ x0)
    # x= S_x x_1 + S_u u
    x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1, cfg.nbVar))
    return u_hat, x_hat


def uni(via_points_raw: np.ndarray, cfg: DictConfig = None):
    print('\033[1;32m--------{}--------\033[0m'.format('Planning smooth trajectory via LQT'))

    cfg = get_config("./", "lqt") if cfg is None else cfg

    via_points = np.zeros((len(via_points_raw), cfg.nbVar))
    via_points[:, :cfg.nbVarPos] = via_points_raw
    start_pose = via_points[0]
    via_point_pose = via_points[1:]

    mu, Q, R, idx_slices, tl = get_matrices(cfg, via_point_pose)
    Su, Sx = set_dynamical_system(cfg)
    u_hat, x_hat = get_u_x(cfg, start_pose, mu, Q, R, Su, Sx)
    return u_hat, x_hat, mu, idx_slices


def uni_hierarchical(via_points_raw: np.ndarray, cfg: DictConfig = None, interval: int = 3):
    print('\033[1;32m--------{}--------\033[0m'.format('Planning smooth trajectory via LQT hierarchically'))

    cfg = get_config("./", "lqt") if cfg is None else cfg

    via_points = np.zeros((len(via_points_raw), cfg.nbVar))
    via_points[:, :cfg.nbVarPos] = via_points_raw[:, :cfg.nbVarPos]
    start_pose = via_points[0]

    x_hat_lst = []
    for i in tqdm(range(0, len(via_points), interval)):
        via_point_pose = via_points[i + 1:i + interval + 1]

        mu, Q, R, idx_slices, tl = get_matrices(cfg, via_point_pose)
        Su, Sx = set_dynamical_system(cfg)
        u_hat, x_hat = get_u_x(cfg, start_pose, mu, Q, R, Su, Sx)
        start_pose = x_hat[-1]
        x_hat_lst.append(x_hat)

    x_hat = np.array(x_hat_lst).reshape((-1, cfg.nbVar))
    return u_hat, x_hat, mu, idx_slices


def bi(via_points_raw_l: np.ndarray, via_points_raw_r: np.ndarray, cfg: DictConfig = None):
    print('\033[1;32m--------{}--------\033[0m'.format('Planning smooth bimanual trajectory via LQT'))

    cfg = get_config("./", "lqt") if cfg is None else cfg

    via_points_l = np.zeros((len(via_points_raw_l), cfg.nbVar))
    via_points_l[:, :cfg.nbVarPos] = via_points_raw_l
    via_points_r = np.zeros((len(via_points_raw_r), cfg.nbVar))
    via_points_r[:, :cfg.nbVarPos] = via_points_raw_r
    l_start_pose = via_points_l[0]
    r_start_pose = via_points_r[0]
    via_point_pose_l = via_points_l[1:]
    via_point_pose_r = via_points_r[1:]

    mu_l, Q, R, idx_slices, tl = get_matrices(cfg, via_point_pose_l)
    mu_r, Q, R, idx_slices, tl = get_matrices(cfg, via_point_pose_r)

    Su, Sx = set_dynamical_system(cfg)

    u_hat_l, x_hat_l = get_u_x(cfg, l_start_pose, mu_l, Q, R, Su, Sx)
    u_hat_r, x_hat_r = get_u_x(cfg, r_start_pose, mu_r, Q, R, Su, Sx)
    return u_hat_l, u_hat_r, x_hat_l, x_hat_r, mu_l, mu_r, idx_slices
