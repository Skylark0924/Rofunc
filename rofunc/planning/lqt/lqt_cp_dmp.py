"""
    Linear Quadratic tracker with control primitives applied on a via-point example

    Refers to https://gitlab.idiap.ch/rli/robotics-codes-from-scratch by Dr. Sylvain Calinon
"""

from math import factorial
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import rofunc as rf
from rofunc.planning.lqt.lqt_cp import define_control_primitive
from rofunc.config.get_config import *
from scipy.linalg import block_diag


def get_matrices(cfg: DictConfig, Mu: np.ndarray):
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
             np.append(-Mu[:, t].reshape(1, -1), 1))) \
                                                 @ block_diag((Q[id0[0]:id0[-1] + 1, id0[0]:id0[-1] + 1]),
                                                              1) @ np.vstack(
            (np.hstack((np.identity(cfg.nbVar), -Mu[:, t].reshape(-1, 1))),
             np.append(np.zeros((1, cfg.nbVar)), 1)))

    Rm = np.identity((cfg.nbData - 1) * cfg.nbVarU) * cfg.rfactor
    return Qm, Rm


def set_dynamical_system(cfg: DictConfig = None):
    A1d = np.zeros(cfg.nbDeriv)
    for i in range(cfg.nbDeriv):
        A1d = A1d + np.diag(np.ones((1, cfg.nbDeriv - i)).flatten(), i) * cfg.dt ** i * 1 / factorial(i)  # Discrete 1D

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
    Su = np.zeros((cfg.nbVarX * cfg.nbData, cfg.nbVarU * (cfg.nbData - 1)))  # It's maybe n-1 not sure
    M = B
    for i in range(1, cfg.nbData):
        Sx[i * cfg.nbVarX:cfg.nbData * cfg.nbVarX, :] = np.dot(
            Sx[i * cfg.nbVarX:cfg.nbData * cfg.nbVarX, :], A)
        Su[cfg.nbVarX * i:cfg.nbVarX * i + M.shape[0], 0:M.shape[1]] = M
        M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]
    return Su, Sx, A, B


def get_u_x(cfg: DictConfig, state_noise: np.ndarray, Mu: np.ndarray, Qm: np.ndarray, Rm: np.ndarray, Su: np.ndarray,
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
    u_hat = np.zeros((2, cfg.nbVarU, cfg.nbData - 1))
    for n in range(2):
        x = np.append(Mu[:, 0] + np.append(np.array([2, 1]), np.zeros(cfg.nbVar - 2)), 1).reshape(-1, 1)
        for t in range(cfg.nbData - 1):
            # Feedback control on augmented state (resulting in feedback and feedforward terms on state)
            u = -Ka[t, :, :] @ x
            x = A @ x + B @ u  # Update of state vector
            if t == 24 and n == 1:
                x = x + state_noise  # Simulated noise on the state
            x_hat[n, :, t] = x.flatten()  # State
    return u_hat, x_hat


def uni_cp_dmp(data: np.ndarray, cfg: DictConfig = None, for_test=False):
    print(
        '\033[1;32m--------{}--------\033[0m'.format('Planning smooth trajectory via LQT (control primitive and DMP)'))
    Qm, Rm = get_matrices(cfg, data)
    PSI, phi = define_control_primitive(cfg)
    Su, Sx, A, B = set_dynamical_system(cfg)

    # state_noise = np.hstack(
    #     (-1, -.2, 1, 0, 0, 0, 0, np.zeros(cfg.nbVarX - cfg.nbVarPos))).reshape(
    #     (cfg.nbVarX, 1))  # Simulated noise on 3d state
    state_noise = np.vstack((np.array([[3], [-0.5]]), np.zeros((cfg.nbVarX-cfg.nbVarU, 1)))) # Simulated noise on 2d state

    u_hat, x_hat = get_u_x(cfg, state_noise, data, Qm, Rm, Su, Sx, PSI, A, B)

    # vis(param, x_hat, u_hat, Mu, idx_slices, tl, phi)
    # vis3d(data, x_hat)
    # rf.visualab.traj_plot([x_hat[:, :2]])
    vis(x_hat, data, for_test=for_test)
    return u_hat, x_hat


def vis(x_hat, Mu, for_test):
    plt.figure()
    plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    plt.plot(Mu[0, :], Mu[1, :], c='blue', linestyle='-', linewidth=2)
    plt.scatter(Mu[0, -1], Mu[1, -1], c='red', s=100)
    plt.scatter(x_hat[0, 0, 0], x_hat[0, 1, 0], c='black', s=50)
    plt.plot(x_hat[0, 0, :], x_hat[0, 1, :], c='black', linestyle=':', linewidth=2)
    plt.plot(x_hat[1, 0, :], x_hat[1, 1, :], c='black', linestyle='-', linewidth=2)
    plt.scatter(x_hat[1, 0, 23:25], x_hat[1, 1, 23:25], c='green', s=30)

    if not for_test:
        plt.show()


def vis3d(data, x_hat):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d', fc='white')

    rf.visualab.traj_plot([x_hat.transpose(0, 2, 1)[0]], mode='3d', ori=False, g_ax=ax, title='Trajectory 1')
    rf.visualab.traj_plot([x_hat.transpose(0, 2, 1)[1]], mode='3d', ori=False, g_ax=ax, title='Trajectory 2')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=20 * 1.5 ** 2, marker='o', color="red", label="Via-points")
    plt.legend()
    plt.show()
