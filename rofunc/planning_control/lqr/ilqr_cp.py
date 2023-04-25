"""
    iLQR for a viapoints task (control primitive version)

    Refers to https://gitlab.idiap.ch/rli/robotics-codes-from-scratch by Dr. Sylvain Calinon
"""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import rofunc as rf

from rofunc.config.utils import get_config
from rofunc.planning_control.lqr.ilqr import fk, f_reach, get_matrices, set_dynamical_system
from rofunc.planning_control.lqr.ilqr_com import fkin0
from rofunc.planning_control.lqt.lqt_cp import define_control_primitive


def get_u_x(cfg: DictConfig, Mu: np.ndarray, Rot: np.ndarray, u: np.ndarray, x0: np.ndarray, Q: np.ndarray,
            R: np.ndarray, Su0: np.ndarray, Sx0: np.ndarray, idx: np.ndarray, tl: np.ndarray, PSI):
    Su = Su0[idx.flatten()]  # We remove the lines that are out of interest

    for i in range(cfg.nbIter):
        x = np.real(Su0 @ u + Sx0 @ x0)
        x = x.reshape([cfg.nbData, cfg.nbVarX])

        f, J = f_reach(cfg, x[tl], Mu, Rot)
        dw = np.linalg.inv(PSI.T @ Su.T @ J.T @ Q @ J @ Su @ PSI + PSI.T @ R @ PSI) @ (
                -PSI.T @ Su.T @ J.T @ Q @ f.flatten() - PSI.T @ u * cfg.rfactor)
        du = PSI @ dw
        # Perform line search
        alpha = 1
        cost0 = f.flatten() @ Q @ f.flatten() + np.linalg.norm(u) * cfg.rfactor

        while True:
            utmp = u + du * alpha
            xtmp = np.real(Su0 @ utmp + Sx0 @ x0)
            xtmp = xtmp.reshape([cfg.nbData, cfg.nbVarX])
            ftmp, _ = f_reach(cfg, xtmp[tl], Mu, Rot)
            cost = ftmp.flatten() @ Q @ ftmp.flatten() + np.linalg.norm(utmp) * cfg.rfactor

            if cost < cost0 or alpha < 1e-3:
                u = utmp
                print("Iteration {}, cost: {}, alpha: {}".format(i, cost, alpha))
                break

            alpha /= 2

        if np.linalg.norm(alpha * du) < 1e-2:
            break
    return u, x


def uni_cp(Mu, Rot, u0, x0, cfg, for_test=False):
    Q, R, idx, tl = get_matrices(cfg)
    PSI, phi = define_control_primitive(cfg)
    Su0, Sx0 = set_dynamical_system(cfg)

    u, x = get_u_x(cfg, Mu, Rot, u0, x0, Q, R, Su0, Sx0, idx, tl, PSI)
    vis(cfg, u, x, Mu, Rot, tl, phi, for_test=for_test)


def vis(cfg, u, x, Mu, Rot, tl, phi, for_test):
    plt.figure()
    plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    # Get points of interest
    f = fk(cfg, x)
    f00 = fkin0(cfg, x[0])
    f10 = fkin0(cfg, x[tl[0]])
    fT0 = fkin0(cfg, x[-1])
    u = u.reshape((-1, cfg.nbVarU))

    plt.plot(f00[:, 0], f00[:, 1], c='black', linewidth=5, alpha=.2)
    plt.plot(f10[:, 0], f10[:, 1], c='black', linewidth=5, alpha=.4)
    plt.plot(fT0[:, 0], fT0[:, 1], c='black', linewidth=5, alpha=.6)

    plt.plot(f[:, 0], f[:, 1], c="black", marker="o", markevery=[0] + tl.tolist())  # ,label="Trajectory"2

    # Plot bounding box or via-points
    ax = plt.gca()
    color_map = ['deepskyblue', 'darkorange']
    for t in range(cfg.nbPoints):
        if cfg.useBoundingBox:
            rect_origin = Mu[t, :2] - Rot[t, :, :] @ np.array(cfg.sz)
            rect_orn = Mu[t, -1]
            rect = patches.Rectangle(rect_origin, cfg.sz[0] * 2, cfg.sz[1] * 2, np.degrees(rect_orn),
                                     color=color_map[t])
            ax.add_patch(rect)
        else:
            plt.scatter(Mu[t, 0], Mu[t, 1], s=100, marker='X', c=color_map[t])

    fig, axs = plt.subplots(7, 1)

    axs[0].plot(f[:, 0], c='black')
    axs[0].set_ylabel("$f(x)_1$")
    axs[0].set_xticks([0, cfg.nbData])
    axs[0].set_xticklabels(["0", "T"])
    for i in range(cfg.nbPoints):
        axs[0].scatter(tl[i], Mu[i, 0], c='blue')

    axs[1].plot(f[:, 1], c='black')
    axs[1].set_ylabel("$f(x)_2$")
    axs[1].set_xticks([0, cfg.nbData])
    axs[1].set_xticklabels(["0", "T"])
    for i in range(cfg.nbPoints):
        axs[1].scatter(tl[i], Mu[i, 1], c='blue')

    axs[2].plot(f[:, 2], c='black')
    axs[2].set_ylabel("$f(x)_3$")
    axs[2].set_xticks([0, cfg.nbData])
    axs[2].set_xticklabels(["0", "T"])
    for i in range(cfg.nbPoints):
        axs[2].scatter(tl[i], Mu[i, 2], c='blue')

    axs[3].plot(u[:, 0], c='black')
    axs[3].set_ylabel("$u_1$")
    axs[3].set_xticks([0, cfg.nbData - 1])
    axs[3].set_xticklabels(["0", "T-1"])

    axs[4].plot(u[:, 1], c='black')
    axs[4].set_ylabel("$u_2$")
    axs[4].set_xticks([0, cfg.nbData - 1])
    axs[4].set_xticklabels(["0", "T-1"])

    axs[5].plot(u[:, 2], c='black')
    axs[5].set_ylabel("$u_3$")
    axs[5].set_xticks([0, cfg.nbData - 1])
    axs[5].set_xticklabels(["0", "T-1"])

    axs[6].set_ylabel("$\phi_k$")
    axs[6].set_xticks([0, cfg.nbData - 1])
    axs[6].set_xticklabels(["0", "T-1"])
    for i in range(cfg.nbFct):
        axs[6].plot(phi[:, i])
    axs[6].set_xlabel("$t$")

    if not for_test:
        plt.show()
