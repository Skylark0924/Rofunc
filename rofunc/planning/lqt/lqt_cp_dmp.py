"""
    Linear Quadratic tracker with control primitives applied on a via-point example
"""

from math import factorial
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rofunc as rf
from scipy.linalg import block_diag
from rofunc.config.get_config import *



def get_matrices(cfg: DictConfig, data: np.ndarray):
    # Task setting (tracking of acceleration profile and reaching of an end-point)
    Q = np.kron(np.identity(cfg.nbData),
                np.diag(np.concatenate((np.zeros((cfg.nbVarU * 2)), np.ones(cfg.nbVarU) * 1e-6))))

    Q[-1 - cfg.nbVar + 1:-1 - cfg.nbVar + 2 * cfg.nbVarU + 1,
    -1 - cfg.nbVar + 1:-1 - cfg.nbVar + 2 * cfg.nbVarU + 1] = np.identity(2 * cfg.nbVarU) * 1e0

    # Weighting matrices in augmented state form
    Qm = np.zeros((cfg.nbVarX * cfg.nbData, cfg.nbVarX * cfg.nbData))

    for t in range(cfg.nbData):
        id0 = np.linspace(0, cfg.nbVar - 1, cfg.nbVar, dtype=int) + t * cfg.nbVar
        id = np.linspace(0, cfg.nbVarX - 1, cfg.nbVarX, dtype=int) + t * cfg.nbVarX
        Qm[id[0]:id[-1] + 1, id[0]:id[-1] + 1] = np.vstack(
            (np.hstack((np.identity(cfg.nbVar), np.zeros((cfg.nbVar, 1)))),
             np.append(-data[:, t].reshape(1, -1), 1))) \
                                                 @ block_diag((Q[id0[0]:id0[-1] + 1, id0[0]:id0[-1] + 1]),
                                                              1) @ np.vstack(
            (np.hstack((np.identity(cfg.nbVar), -data[:, t].reshape(-1, 1))),
             np.append(np.zeros((1, cfg.nbVar)), 1)))

    Rm = np.identity((cfg.nbData - 1) * cfg.nbVarU) * cfg.rfactor
    return Qm, Rm


def set_dynamical_system(cfg: DictConfig = None):
    A1d = np.zeros(cfg.nbDeriv)
    for i in range(cfg.nbDeriv):
        A1d = A1d + np.diag(np.ones((1, cfg.nbDeriv - i)).flatten(), i) * cfg.dt ** i * 1 / factorial(
            i)  # Discrete 1D

    B1d = np.zeros((cfg.nbDeriv, 1))
    for i in range(cfg.nbDeriv):
        B1d[cfg.nbDeriv - 1 - i] = cfg.dt ** (i + 1) * 1 / factorial(i + 1)  # Discrete 1D

    A0 = np.kron(A1d, np.eye(cfg.nbVarU))  # Discrete nD
    B0 = np.kron(B1d, np.eye(cfg.nbVarU))  # Discrete nD

    A = np.vstack((np.hstack((A0, np.zeros((cfg.nbVar, 1)))),
                   np.hstack((np.zeros((cfg.nbVar)), 1)).reshape(1, -1)))  # Augmented A (homogeneous)
    B = np.vstack((B0, np.zeros((1, cfg.nbVarU))))  # Augmented B (homogeneous)

    # Build Sx and Su transfer matrices (for augmented state space)
    Sx = np.kron(np.ones((cfg.nbData, 1)), np.eye(cfg.nbVarX, cfg.nbVarX))
    Su = np.zeros(
        (cfg.nbVarX * cfg.nbData, cfg.nbVarU * (cfg.nbData - 1)))  # It's maybe n-1 not sure
    M = B
    for i in range(1, cfg.nbData):
        Sx[i * cfg.nbVarX:cfg.nbData * cfg.nbVarX, :] = np.dot(
            Sx[i * cfg.nbVarX:cfg.nbData * cfg.nbVarX, :], A)
        Su[cfg.nbVarX * i:cfg.nbVarX * i + M.shape[0], 0:M.shape[1]] = M
        M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]
    return Su, Sx, A, B


def get_u_x(cfg: DictConfig, state_noise: np.ndarray, muQ: np.ndarray, Qm: np.ndarray, Rm: np.ndarray, Su: np.ndarray,
            Sx: np.ndarray, PSI: np.ndarray, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Least squares formulation of recursive LQR with an augmented state space and control primitives
    W = np.linalg.inv(PSI.T @ Su.T @ Qm @ Su @ PSI + PSI.T @ Rm @ PSI) @ PSI.T @ Su.T @ Qm @ Sx
    F = PSI @ W  # F with control primitives

    # Reproduction with feedback controller on augmented state space (with CP)
    Ka = np.empty((cfg.nbData - 1, cfg.nbVarU, cfg.nbVarX))
    Ka[0, :, :] = F[0:cfg.nbVarU, :]
    P = np.identity(cfg.nbVarX)
    for t in range(cfg.nbData - 2):
        id = t * cfg.nbVarU + np.linspace(2, cfg.nbVarU + 1, cfg.nbVarU, dtype=int)
        P = P @ np.linalg.pinv(A - B @ Ka[t, :, :])
        Ka[t + 1, :, :] = F[id, :] @ P

    x_hat = np.zeros((2, cfg.nbVarX, cfg.nbData - 1))
    u_hat = np.zeros((2, cfg.nbVarPos, cfg.nbData - 1))
    for n in range(2):
        x = np.append(muQ[:, 0] + np.append(np.array([2, 1]), np.zeros(cfg.nbVar - 2)), 1).reshape(-1, 1)
        for t in range(cfg.nbData - 1):
            # Feedback control on augmented state (resulting in feedback and feedforward terms on state)
            u = -Ka[t, :, :] @ x
            x = A @ x + B @ u  # Update of state vector
            if t == 24 and n == 1:
                x = x + state_noise  # Simulated noise on the state
            x_hat[n, :, t] = x.flatten()  # State
    return u_hat, x_hat


def uni_cp_dmp(data: np.ndarray, cfg: DictConfig = None):
    print(
        '\033[1;32m--------{}--------\033[0m'.format('Planning smooth trajectory via LQT (control primitive and DMP)'))
    # data = data[:, :cfg.nbVarPos]

    # start_pose = np.zeros((param['nbVar'],), dtype=np.float32)
    # start_pose[:cfg.nbVarPos] = data[0]

    # via_point_pose = data[1:]
    # cfg.nbPoints = len(via_point_pose)

    Qm, Rm = get_matrices(cfg, data)
    PSI, phi = rf.lqt.define_control_primitive(cfg)
    Su, Sx, A, B = set_dynamical_system(cfg)

    state_noise = np.hstack(
        (-1, -.2, 1, 0, 0, 0, 0, np.zeros(cfg.nbVarX - cfg.nbVarPos))).reshape((cfg.nbVarX, 1))  # Simulated noise on state

    u_hat, x_hat = get_u_x(cfg, state_noise, data, Qm, Rm, Su, Sx, PSI, A, B)

    # vis(param, x_hat, u_hat, muQ, idx_slices, tl, phi)
    vis3d(data, x_hat)
    # rf.visualab.traj_plot([x_hat[:, :2]])
    # vis(x_hat, data)
    return u_hat, x_hat


def vis(x_hat, muQ):
    plt.figure()
    plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    plt.plot(muQ[0, :], muQ[1, :], c='blue', linestyle='-', linewidth=2)
    plt.scatter(muQ[0, -1], muQ[1, -1], c='red', s=100)
    plt.scatter(x_hat[0, 0, 0], x_hat[0, 1, 0], c='black', s=50)
    plt.plot(x_hat[0, 0, :], x_hat[0, 1, :], c='black', linestyle=':', linewidth=2)
    plt.plot(x_hat[1, 0, :], x_hat[1, 1, :], c='black', linestyle='-', linewidth=2)
    plt.scatter(x_hat[1, 0, 23:25], x_hat[1, 1, 23:25], c='green', s=30)

    plt.show()


def vis3d(data, x_hat):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d', fc='white')

    rf.visualab.traj_plot([x_hat.transpose(0, 2, 1)[0]], mode='3d', ori=False, g_ax=ax, title='Trajectory 1')
    rf.visualab.traj_plot([x_hat.transpose(0, 2, 1)[1]], mode='3d', ori=False, g_ax=ax, title='Trajectory 2')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=20 * 1.5 ** 2, marker='o', color="red", label="Via-points")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    cfg = get_config('./', 'lqt')

    data_raw = np.load('/home/ubuntu/Data/2022_09_09_Taichi/rep3_r.npy')
    # filter_indices = [i for i in range(0, len(data_raw) - 10, 5)]
    # filter_indices.append(len(data_raw) - 1)
    MuPos = data_raw  # pose
    MuVel = np.gradient(MuPos)[0] / cfg.dt
    MuAcc = np.gradient(MuVel)[0] / cfg.dt
    via_points = np.hstack((MuPos, MuVel, MuAcc)).T
    cfg.nbData = len(via_points[0])
    # 2d letter example data
    # from scipy.interpolate import interp1d
    # x = np.load(
    #     '/home/ubuntu/Github/Knowledge-Universe/Robotics/Roadmap-for-robot-science/rofunc/planning/src/robotics-codes-from-scratch-master/data/2Dletters/S.npy')[
    #     0, :, :2].T
    #
    # f_pos = interp1d(np.linspace(0, np.size(x, 1) - 1, np.size(x, 1), dtype=int), x, kind='cubic')
    # MuPos = f_pos(np.linspace(0, np.size(x, 1) - 1, cfg.nbData))  # Position
    # MuVel = np.gradient(MuPos)[1] / cfg.dt
    # MuAcc = np.gradient(MuVel)[1] / cfg.dt
    # # Position, velocity and acceleration profiles as references
    # via_points = np.vstack((MuPos, MuVel, MuAcc))

    uni_cp_dmp(via_points, cfg)
