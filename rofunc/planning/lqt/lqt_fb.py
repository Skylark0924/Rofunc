"""
    LQT computed in a recursive way (via-point example)
"""
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import rofunc as rf


def get_matrices(param: Dict, Mu: np.ndarray):
    param['nbPoints'] = len(Mu)

    R = np.eye(param["nbVarPos"]) * param["rfactor"]  # Control cost matrix

    # Sparse reference with a set of via-points
    tl = np.linspace(0, param["nbData"] - 1, param["nbPoints"] + 1)
    tl = np.rint(tl[1:])

    # Definition of augmented precision matrix Qa based on standard precision matrix Q0
    Q0 = np.diag(np.hstack([np.ones(param["nbVarPos"]), np.zeros(param["nbVar"] - param["nbVarPos"])]))

    Q0_augmented = np.identity(param["nbVar"] + 1)
    Q0_augmented[:param["nbVar"], :param["nbVar"]] = Q0

    Q = np.zeros([param["nbVar"] + 1, param["nbVar"] + 1, param["nbData"]])
    for i in range(param["nbPoints"]):
        Q[:, :, int(tl[i])] = np.vstack([
            np.hstack([np.identity(param["nbVar"]), np.zeros([param["nbVar"], 1])]),
            np.hstack([-Mu[i, :], 1])]) @ Q0_augmented @ np.vstack([
            np.hstack([np.identity(param["nbVar"]), -Mu[i, :].reshape([-1, 1])]),
            np.hstack([np.zeros(param["nbVar"]), 1])])
    return Q, R, tl


def set_dynamical_system(param: Dict):
    """
    Dynamical System settings (discrete version)
    Args:
        param:

    Returns:

    """
    A1d = np.zeros((param["nbDeriv"], param["nbDeriv"]))
    for i in range(param["nbDeriv"]):
        A1d += np.diag(np.ones((param["nbDeriv"] - i,)), i) * param["dt"] ** i / np.math.factorial(i)  # Discrete 1D

    B1d = np.zeros((param["nbDeriv"], 1))
    for i in range(0, param["nbDeriv"]):
        B1d[param["nbDeriv"] - i - 1, :] = param["dt"] ** (i + 1) * 1 / np.math.factorial(i + 1)  # Discrete 1D

    A0 = np.kron(A1d, np.eye(param["nbVarPos"]))  # Discrete nD
    B0 = np.kron(B1d, np.eye(param["nbVarPos"]))  # Discrete nD
    A = np.eye(A0.shape[0] + 1)  # Augmented A
    A[:A0.shape[0], :A0.shape[1]] = A0
    B = np.vstack((B0, np.zeros((1, param["nbVarPos"]))))  # Augmented B
    return A, B


def get_u_x(param: Dict, state_noise: np.ndarray, P: np.ndarray, R: np.ndarray, A: np.ndarray, B: np.ndarray):
    """
    Reproduction with only feedback (FB) on augmented state
    Args:
        param:
        state_noise:
        P:
        R:
        A:
        B:

    Returns:

    """
    x_hat = np.zeros((param["nbVar"] + 1, 2, param["nbData"]))
    u_hat = np.zeros((param["nbVarPos"], 2, param["nbData"]))
    for n in range(2):
        x = np.hstack([np.zeros(param["nbVar"]), 1])
        for t in range(param["nbData"]):
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


def uni_fb(param: Dict, data: np.ndarray):
    """
    LQR with recursive computation and augmented state space
    Args:
        param:
        data:

    Returns:

    """
    Q, R, tl = get_matrices(param, data)
    A, B = set_dynamical_system(param)

    state_noise = np.hstack((-1, -.2, 1, 0, 0, 0, 0, np.zeros(param["nbVar"] + 1 - param["nbVarPos"])))

    P = np.zeros((param["nbVarX"], param["nbVarX"], param["nbData"]))
    P[:, :, -1] = Q[:, :, -1]

    for t in range(param["nbData"] - 2, -1, -1):
        P[:, :, t] = Q[:, :, t] - A.T @ (
                P[:, :, t + 1] @ np.dot(B, np.linalg.pinv(B.T @ P[:, :, t + 1] @ B + R))
                @ B.T @ P[:, :, t + 1] - P[:, :, t + 1]) @ A
    u_hat, x_hat = get_u_x(param, state_noise, P, R, A, B)
    vis3d(data, x_hat)
    return u_hat, x_hat


def vis(data, r):
    plt.figure()
    for n in range(2):
        plt.plot(r[0, n, :], r[1, n, :], label="Trajectory {}".format(n + 1))
        plt.scatter(r[0, n, 0], r[1, n, 0], marker='o')

    plt.scatter(data[:, 0], data[:, 1], s=20 * 1.5 ** 2, marker='o', color="red", label="Via-points")
    plt.legend()
    plt.show()


def vis3d(data, r):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d', fc='white')

    rf.visualab.traj_plot([r.transpose(1, 2, 0)[0]], mode='3d', ori=False, g_ax=ax, title='Trajectory 1')
    rf.visualab.traj_plot([r.transpose(1, 2, 0)[1]], mode='3d', ori=False, g_ax=ax, title='Trajectory 2')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=20 * 1.5 ** 2, marker='o', color="red", label="Via-points")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    param = {
        "nbData": 100,  # Number of datapoints
        "nbVarPos": 7,  # Dimension of position data (here: x1,x2)
        "nbDeriv": 2,  # Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
        "dt": 1E-2,  # Time step duration
        "rfactor": 1E-6,  # control cost in LQR
    }
    param["nbVar"] = param["nbVarPos"] * param["nbDeriv"]  # Dimension of state vector
    param["nbVarX"] = param["nbVar"] + 1  # Augmented state space

    via_points = np.zeros((2, 14))
    via_points[0, :7] = np.array([2, 5, 3, 0, 0, 0, 1])
    via_points[1, :7] = np.array([3, 1, 1, 0, 0, 0, 1])

    # via_points = np.array([[2, 5, 0, 0], [3, 1, 0, 0]])

    uni_fb(param, via_points)
