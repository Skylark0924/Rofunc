"""
    LQT computed in a recursive way (via-point example)

    Refers to https://gitlab.idiap.ch/rli/robotics-codes-from-scratch by Dr. Sylvain Calinon
"""
import matplotlib.pyplot as plt
import numpy as np

import rofunc as rf
from rofunc.config.get_config import *


def get_matrices(cfg: DictConfig, via_points: np.ndarray):
    cfg.nbPoints = len(via_points)

    # Control cost matrix
    R = np.eye(cfg.nbVarPos) * cfg.rfactor

    # Sparse reference with a set of via-points
    tl = np.linspace(0, cfg.nbData - 1, cfg.nbPoints + 1)
    tl = np.rint(tl[1:])

    # Definition of augmented precision matrix Qa based on standard precision matrix Q0
    Q0 = np.diag(np.hstack([np.ones(cfg.nbVarPos), np.zeros(cfg.nbVar - cfg.nbVarPos)]))
    Q0_augmented = np.identity(cfg.nbVarX)
    Q0_augmented[:cfg.nbVar, :cfg.nbVar] = Q0

    Q = np.zeros([cfg.nbVarX, cfg.nbVarX, cfg.nbData])
    for i in range(cfg.nbPoints):
        Q[:, :, int(tl[i])] = np.vstack([
            np.hstack([np.identity(cfg.nbVar), np.zeros([cfg.nbVar, 1])]),
            np.hstack([-via_points[i, :], 1])]) @ Q0_augmented @ np.vstack([
            np.hstack([np.identity(cfg.nbVar), -via_points[i, :].reshape([-1, 1])]),
            np.hstack([np.zeros(cfg.nbVar), 1])])
    return Q, R, tl


def set_dynamical_system(cfg: DictConfig):
    A1d = np.zeros((cfg.nbDeriv, cfg.nbDeriv))
    for i in range(cfg.nbDeriv):
        A1d += np.diag(np.ones((cfg.nbDeriv - i,)), i) * cfg.dt ** i / np.math.factorial(i)  # Discrete 1D

    B1d = np.zeros((cfg.nbDeriv, 1))
    for i in range(0, cfg.nbDeriv):
        B1d[cfg.nbDeriv - i - 1, :] = cfg.dt ** (i + 1) * 1 / np.math.factorial(i + 1)  # Discrete 1D

    A0 = np.kron(A1d, np.eye(cfg.nbVarPos))  # Discrete nD
    B0 = np.kron(B1d, np.eye(cfg.nbVarPos))  # Discrete nD
    A = np.eye(A0.shape[0] + 1)  # Augmented A
    A[:A0.shape[0], :A0.shape[1]] = A0
    B = np.vstack((B0, np.zeros((1, cfg.nbVarPos))))  # Augmented B
    return A, B


def get_u_x(cfg: DictConfig, state_noise: np.ndarray, P: np.ndarray, R: np.ndarray, A: np.ndarray, B: np.ndarray):
    """
    Reproduction with only feedback (FB) on augmented state
    Args:
        cfg:
        state_noise:
        P:
        R:
        A:
        B:

    Returns:

    """
    x_hat = np.zeros((cfg.nbVarX, 2, cfg.nbData))
    u_hat = np.zeros((cfg.nbVarPos, 2, cfg.nbData))
    for n in range(2):
        x = np.hstack([np.zeros(cfg.nbVar), 1])
        for t in range(cfg.nbData):
            Z_bar = B.T @ P[:, :, t] @ B + R
            K = np.linalg.inv(Z_bar.T @ Z_bar) @ Z_bar.T @ B.T @ P[:, :, t] @ A  # Feedback gain
            u = -K @ x  # Acceleration command with FB on augmented state (resulting in feedback and feedforward terms)
            x = A @ x + B @ u  # Update of state vector

            if t == 25 and n == 1:
                x += state_noise

            if t == 70 and n == 1:
                x += state_noise

            if t == 75 and n == 1:
                x += state_noise

            x_hat[:, n, t] = x  # Log data
            u_hat[:, n, t] = u  # Log data
    return u_hat, x_hat


def uni_fb(via_points: np.ndarray, cfg: DictConfig = None, for_test=False):
    """
    LQR with recursive computation and augmented state space
    Args:
        via_points:
        cfg:
        for_test:

    Returns:

    """
    print('\033[1;32m--------{}--------\033[0m'.format('LQT with feedback control'))

    cfg = get_config("./", "lqt") if cfg is None else cfg
    # cfg.nbVarPos = 2

    Q, R, tl = get_matrices(cfg, via_points)
    A, B = set_dynamical_system(cfg)

    state_noise = np.hstack(
        (-1, -.2, 1, 0, 0, 0, 0, np.zeros(cfg.nbVarX - cfg.nbVarPos)))  # TODO: real noise from robot
    # state_noise = np.hstack(
    #     (-1, -.2, np.zeros(cfg.nbVarX - cfg.nbVarPos)))

    P = np.zeros((cfg.nbVarX, cfg.nbVarX, cfg.nbData))
    P[:, :, -1] = Q[:, :, -1]

    for t in range(cfg.nbData - 2, -1, -1):
        P[:, :, t] = Q[:, :, t] - A.T @ (
                P[:, :, t + 1] @ np.dot(B, np.linalg.pinv(B.T @ P[:, :, t + 1] @ B + R))
                @ B.T @ P[:, :, t + 1] - P[:, :, t + 1]) @ A
    u_hat, x_hat = get_u_x(cfg, state_noise, P, R, A, B)
    vis3d(via_points, x_hat, for_test=for_test)
    return u_hat, x_hat


def vis(via_points, x_hat):
    plt.figure()
    for n in range(2):
        plt.plot(x_hat[0, n, :], x_hat[1, n, :], label="Trajectory {}".format(n + 1))
        plt.scatter(x_hat[0, n, 0], x_hat[1, n, 0], marker='o')

    plt.scatter(via_points[:, 0], via_points[:, 1], s=20 * 1.5 ** 2, marker='o', color="red", label="Via-points")
    plt.legend()
    plt.show()


def vis3d(via_points, x_hat, for_test):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d', fc='white')

    rf.visualab.traj_plot([x_hat.transpose(1, 2, 0)[0]], mode='3d', ori=False, g_ax=ax, legend='Trajectory 1')
    rf.visualab.traj_plot([x_hat.transpose(1, 2, 0)[1]], mode='3d', ori=False, g_ax=ax, legend='Trajectory 2')

    ax.scatter(via_points[:, 0], via_points[:, 1], via_points[:, 2], s=20 * 1.5 ** 2, marker='o', color="red",
               label="Via-points")
    plt.legend()
    if not for_test:
        plt.show()
