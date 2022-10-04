"""
    Linear Quadratic tracker with control primitives applied on a via-point example
"""

import numpy as np
import rofunc as rf
import matplotlib.pyplot as plt
from scipy import special
from math import factorial
from typing import Union, Dict, Tuple


# Building piecewise constant basis functions
def build_phi_piecewise(nb_data, nb_fct):
    phi = np.kron(np.identity(nb_fct), np.ones((int(np.ceil(nb_data / nb_fct)), 1)))
    return phi[:nb_data]


# Building radial basis functions (RBFs)
def build_phi_rbf(nb_data, nb_fct):
    t = np.linspace(0, 1, nb_data).reshape((-1, 1))
    tMu = np.linspace(t[0], t[-1], nb_fct)
    phi = np.exp(-1e2 * (t.T - tMu) ** 2)
    return phi.T


# Building Bernstein basis functions
def build_phi_bernstein(nb_data, nb_fct):
    t = np.linspace(0, 1, nb_data)
    phi = np.zeros((nb_data, nb_fct))
    for i in range(nb_fct):
        phi[:, i] = factorial(nb_fct - 1) / (factorial(i) * factorial(nb_fct - 1 - i)) * (1 - t) ** (
                nb_fct - 1 - i) * t ** i
    return phi


# Building Fourier basis functions
def build_phi_fourier(nb_data, nb_fct):
    t = np.linspace(0, 1, nb_data).reshape((-1, 1))

    # Alternative computation for real and even signal
    k = np.arange(0, nb_fct).reshape((-1, 1))
    phi = np.cos(t.T * k * 2 * np.pi)
    return phi.T


# Dynamical System settings (discrete)
# =====================================
def set_dynamical_system(param: Dict):
    nb_var = param["nb_var"]
    A = np.identity(nb_var)
    if param["nbDeriv"] == 2:
        A[:param["nbVarPos"], -param["nbVarPos"]:] = np.identity(param["nbVarPos"]) * param["dt"]

    B = np.zeros((nb_var, param["nbVarPos"]))
    derivatives = [param["dt"], param["dt"] ** 2 / 2][:param["nbDeriv"]]
    for i in range(param["nbDeriv"]):
        B[i * param["nbVarPos"]:(i + 1) * param["nbVarPos"]] = np.identity(param["nbVarPos"]) * derivatives[::-1][i]

    # Build Sx and Su transfer matrices
    Su = np.zeros((nb_var * param["nbData"], param["nbVarPos"] * (param["nbData"] - 1)))  # It's maybe n-1 not sure
    Sx = np.kron(np.ones((param["nbData"], 1)), np.eye(nb_var, nb_var))

    M = B
    for i in range(1, param["nbData"]):
        Sx[i * nb_var:param["nbData"] * nb_var, :] = np.dot(Sx[i * nb_var:param["nbData"] * nb_var, :], A)
        Su[nb_var * i:nb_var * i + M.shape[0], 0:M.shape[1]] = M
        M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]

    return Su, Sx


def get_u_x(param: Dict, start_pose: np.ndarray, muQ: np.ndarray, Q: np.ndarray, R: np.ndarray, Su: np.ndarray,
            Sx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x0 = np.zeros((param["nb_var"], 1))
    w_hat = np.linalg.inv(PSI.T @ Su.T @ Q @ Su @ PSI + PSI.T @ R @ PSI) @ PSI.T @ Su.T @ Q @ (param["muQ"] - Sx @ x0)
    u_hat = PSI @ w_hat
    x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1, param["nb_var"]))
    u_hat = u_hat.reshape((-1, param["nbVarPos"]))
    return u_hat, x_hat


def uni_cp(param: Dict, data: np.ndarray):
    print('\033[1;32m--------{}--------\033[0m'.format('Planning smooth trajectory via LQT'))

    start_pose = np.zeros((14,), dtype=np.float32)
    start_pose[:7] = data[0]

    via_point_pose = data[1:]
    param['nbPoints'] = len(via_point_pose)

    via_point, muQ, Q, R, idx_slices, tl = rf.lqt.get_matrices(param, via_point_pose)
    Su, Sx = set_dynamical_system(param)
    u_hat, x_hat = get_u_x(param, start_pose, muQ, Q, R, Su, Sx)

    vis(x_hat, u_hat, idx_slices, tl)
    return u_hat, x_hat, muQ, idx_slices


def vis(x_hat, u_hat, idx_slices, tl):
    plt.figure()

    plt.title("2D Trajectory")
    plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    plt.scatter(x_hat[0, 0], x_hat[0, 1], c='black', s=100)

    for slice_t in idx_slices:
        plt.scatter(param["muQ"][slice_t][0], param["muQ"][slice_t][1], c='blue', s=100)

    plt.plot(x_hat[:, 0], x_hat[:, 1], c='black')

    fig, axs = plt.subplots(5, 1)

    axs[0].plot(x_hat[:, 0], c='black')
    axs[0].set_ylabel("$x_1$")
    axs[0].set_xticks([0, param["nbData"]])
    axs[0].set_xticklabels(["0", "T"])
    for t in tl:
        axs[0].scatter(t, x_hat[t, 0], c='blue')

    axs[1].plot(x_hat[:, 1], c='black')
    axs[1].set_ylabel("$x_2$")
    axs[1].set_xticks([0, param["nbData"]])
    axs[1].set_xticklabels(["0", "T"])
    for t in tl:
        axs[1].scatter(t, x_hat[t, 1], c='blue')

    axs[2].plot(u_hat[:, 0], c='black')
    axs[2].set_ylabel("$u_1$")
    axs[2].set_xticks([0, param["nbData"] - 1])
    axs[2].set_xticklabels(["0", "T-1"])

    axs[3].plot(u_hat[:, 1], c='black')
    axs[3].set_ylabel("$u_2$")
    axs[3].set_xticks([0, param["nbData"] - 1])
    axs[3].set_xticklabels(["0", "T-1"])

    axs[4].set_ylabel("$\phi_k$")
    axs[4].set_xticks([0, param["nbData"] - 1])
    axs[4].set_xticklabels(["0", "T-1"])
    for i in range(param["nbFct"]):
        axs[4].plot(phi[:, i])
    axs[4].set_xlabel("$t$")

    plt.show()


if __name__ == '__main__':
    param = {
        "nbData": 200,  # Number of data points
        "nbVarPos": 7,  # Dimension of position data
        "nbDeriv": 2,  # Number of static and dynamic features (2 -> [x,dx])
        "dt": 1e-2,  # Time step duration
        "rfactor": 1e-8  # Control cost
    }
    param["nb_var"] = param["nbVarPos"] * param["nbDeriv"]  # Dimension of state vector

    # Building basis functions
    # =========================

    functions = {
        "PIECEWEISE": build_phi_piecewise,
        "RBF": build_phi_rbf,
        "BERNSTEIN": build_phi_bernstein,
        "FOURIER": build_phi_fourier
    }
    phi = functions[param["basisName"]](param["nbData"] - 1, param["nbFct"])
    PSI = np.kron(phi, np.identity(param["nbVarPos"]))

    uni_cp(param)
