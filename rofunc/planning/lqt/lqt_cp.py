"""
    Linear Quadratic tracker with control primitives applied on a via-point example

    Refers to https://gitlab.idiap.ch/rli/robotics-codes-from-scratch by Dr. Sylvain Calinon
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import rofunc as rf
from rofunc.config.get_config import *


def define_control_primitive(cfg):
    functions = {
        "PIECEWISE": rf.primitive.build_phi_piecewise,
        "RBF": rf.primitive.build_phi_rbf,
        "BERNSTEIN": rf.primitive.build_phi_bernstein,
        "FOURIER": rf.primitive.build_phi_fourier
    }
    phi = functions[cfg.basisName](cfg.nbData - 1, cfg.nbFct)
    PSI = np.kron(phi, np.identity(cfg.nbVarPos))
    return PSI, phi


def set_dynamical_system(cfg: DictConfig):
    A = np.identity(cfg.nbVar)
    if cfg.nbDeriv == 2:
        A[:cfg.nbVarPos, -cfg.nbVarPos:] = np.identity(cfg.nbVarPos) * cfg.dt

    B = np.zeros((cfg.nbVar, cfg.nbVarPos))
    derivatives = [cfg.dt, cfg.dt ** 2 / 2][:cfg.nbDeriv]
    for i in range(cfg.nbDeriv):
        B[i * cfg.nbVarPos:(i + 1) * cfg.nbVarPos] = np.identity(cfg.nbVarPos) * derivatives[::-1][i]

    # Build Sx and Su transfer matrices
    Su = np.zeros((cfg.nbVar * cfg.nbData, cfg.nbVarPos * (cfg.nbData - 1)))  # It's maybe n-1 not sure
    Sx = np.kron(np.ones((cfg.nbData, 1)), np.eye(cfg.nbVar, cfg.nbVar))

    M = B
    for i in range(1, cfg.nbData):
        Sx[i * cfg.nbVar:cfg.nbData * cfg.nbVar, :] = np.dot(Sx[i * cfg.nbVar:cfg.nbData * cfg.nbVar, :], A)
        Su[cfg.nbVar * i:cfg.nbVar * i + M.shape[0], 0:M.shape[1]] = M
        M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]
    return Su, Sx


def get_u_x(cfg: DictConfig, start_pose: np.ndarray, muQ: np.ndarray, Q: np.ndarray, R: np.ndarray, Su: np.ndarray,
            Sx: np.ndarray, PSI: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x0 = start_pose.reshape((cfg.nbVar, 1))
    w_hat = np.linalg.inv(PSI.T @ Su.T @ Q @ Su @ PSI + PSI.T @ R @ PSI) @ PSI.T @ Su.T @ Q @ (muQ - Sx @ x0)
    u_hat = PSI @ w_hat
    x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1, cfg.nbVar))
    u_hat = u_hat.reshape((-1, cfg.nbVarPos))
    return u_hat, x_hat


def uni_cp(via_points_raw: np.ndarray, cfg: DictConfig = None):
    print('\033[1;32m--------{}--------\033[0m'.format('Planning smooth trajectory via LQT (control primitive)'))

    cfg = get_config("./", "lqt") if cfg is None else cfg

    via_points = np.zeros((len(via_points_raw), cfg.nbVar))
    via_points[:, :cfg.nbVarPos] = via_points_raw
    start_pose = via_points[0]
    via_point_pose = via_points[1:]

    mu, Q, R, idx_slices, tl = rf.lqt.get_matrices(cfg, via_point_pose)
    PSI, phi = define_control_primitive(cfg)
    Su, Sx = set_dynamical_system(cfg)
    u_hat, x_hat = get_u_x(cfg, start_pose, mu, Q, R, Su, Sx, PSI)

    # vis(param, x_hat, u_hat, muQ, idx_slices, tl, phi)
    # rf.visualab.traj_plot([x_hat[:, :2]])
    return u_hat, x_hat, mu, idx_slices


def vis(cfg, x_hat, u_hat, muQ, idx_slices, tl, phi):
    plt.figure()

    plt.title("2D Trajectory")
    plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    plt.scatter(x_hat[0, 0], x_hat[0, 1], c='black', s=100)

    for slice_t in idx_slices:
        plt.scatter(muQ[slice_t][0], muQ[slice_t][1], c='blue', s=100)

    plt.plot(x_hat[:, 0], x_hat[:, 1], c='black')

    fig, axs = plt.subplots(5, 1)

    axs[0].plot(x_hat[:, 0], c='black')
    axs[0].set_ylabel("$x_1$")
    axs[0].set_xticks([0, cfg.nbData])
    axs[0].set_xticklabels(["0", "T"])
    for t in tl:
        axs[0].scatter(t, x_hat[t, 0], c='blue')

    axs[1].plot(x_hat[:, 1], c='black')
    axs[1].set_ylabel("$x_2$")
    axs[1].set_xticks([0, cfg.nbData])
    axs[1].set_xticklabels(["0", "T"])
    for t in tl:
        axs[1].scatter(t, x_hat[t, 1], c='blue')

    axs[2].plot(u_hat[:, 0], c='black')
    axs[2].set_ylabel("$u_1$")
    axs[2].set_xticks([0, cfg.nbData - 1])
    axs[2].set_xticklabels(["0", "T-1"])

    axs[3].plot(u_hat[:, 1], c='black')
    axs[3].set_ylabel("$u_2$")
    axs[3].set_xticks([0, cfg.nbData - 1])
    axs[3].set_xticklabels(["0", "T-1"])

    axs[4].set_ylabel("$\phi_k$")
    axs[4].set_xticks([0, cfg.nbData - 1])
    axs[4].set_xticklabels(["0", "T-1"])
    for i in range(cfg.nbFct):
        axs[4].plot(phi[:, i])
    axs[4].set_xlabel("$t$")

    plt.show()

