"""
    iLQR for a viapoints task (control primitive version)

    Refers to https://gitlab.idiap.ch/rli/robotics-codes-from-scratch by Dr. Sylvain Calinon
"""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib

from rofunc.config.get_config import *
from rofunc.planning.lqr.ilqr import logmap, get_matrices, set_dynamical_system
from rofunc.planning.lqt.lqt_cp import define_control_primitive


def fkin(cfg, x):
    L = np.tril(np.ones([cfg.nbVarX, cfg.nbVarX]))
    f = np.vstack((cfg.l @ np.cos(L @ x),
                   cfg.l @ np.sin(L @ x),
                   np.mod(np.sum(x, 0) + np.pi,
                          2 * np.pi) - np.pi))  # x1,x2,o (orientation as single Euler angle for planar robot)
    return f


def fkin0(cfg, x):
    T = np.tril(np.ones([cfg.nbVarX, cfg.nbVarX]))
    T2 = np.tril(np.matlib.repmat(cfg.l, len(x), 1))
    f = np.vstack((
        T2 @ np.cos(T @ x),
        T2 @ np.sin(T @ x)
    ))
    f = np.hstack([np.zeros([2, 1]), f])
    return f


def jkin(cfg, x):
    L = np.tril(np.ones([cfg.nbVarX, cfg.nbVarX]))
    J = np.vstack((
        -np.sin(L @ x).T @ np.diag(cfg.l) @ L,
        np.cos(L @ x).T @ np.diag(cfg.l) @ L,
        np.ones([1, cfg.nbVarX])
    ))
    return J


def f_reach(cfg, robot_state, Mu, Rot):
    f = logmap(fkin(cfg, robot_state), Mu)
    J = np.zeros([cfg.nbPoints * cfg.nbVarF, cfg.nbPoints * cfg.nbVarX])

    for t in range(cfg.nbPoints):
        f[:2, t] = Rot[:, :, t].T @ f[:2, t]  # Object oriented fk
        Jtmp = jkin(cfg, robot_state[:, t])
        Jtmp[:2] = Rot[:, :, t].T @ Jtmp[:2]  # Object centered jacobian

        if cfg.useBoundingBox:
            for i in range(2):
                if abs(f[i, t]) < cfg.sz[i]:
                    f[i, t] = 0
                    Jtmp[i] = 0
                else:
                    f[i, t] -= np.sign(f[i, t]) * cfg.sz[i]
        J[t * cfg.nbVarF:(t + 1) * cfg.nbVarF, t * cfg.nbVarX:(t + 1) * cfg.nbVarX] = Jtmp
    return f, J


def get_u_x(cfg: DictConfig, Mu: np.ndarray, Rot: np.ndarray, Q: np.ndarray, R: np.ndarray, Su0: np.ndarray,
            Sx0: np.ndarray, idx: np.ndarray, tl: np.ndarray, PSI):
    Su = Su0[idx.flatten()]  # We remove the lines that are out of interest

    u = np.zeros(cfg.nbVarU * (cfg.nbData - 1))  # Initial control command
    x0 = np.array([3 * np.pi / 4, -np.pi / 2, -np.pi / 4])  # Initial state
    # x0 = fkin(np.array([3 * np.pi / 4, -np.pi / 2, -np.pi / 4]), cfg).reshape((3,))  # Initial state

    for i in range(cfg.nbIter):
        x = np.real(Su0 @ u + Sx0 @ x0)
        x = x.reshape([cfg.nbVarX, cfg.nbData], order='F')

        f, J = f_reach(cfg, x[:, tl], Mu, Rot)
        dw = np.linalg.inv(PSI.T @ Su.T @ J.T @ Q @ J @ Su @ PSI + PSI.T @ R @ PSI) @ (
                -PSI.T @ Su.T @ J.T @ Q @ f.flatten('F') - PSI.T @ u * cfg.rfactor)
        du = PSI @ dw
        # Perform line search
        alpha = 1
        cost0 = f.flatten('F') @ Q @ f.flatten('F') + np.linalg.norm(u) * cfg.rfactor

        while True:
            utmp = u + du * alpha
            xtmp = np.real(Su0 @ utmp + Sx0 @ x0)
            xtmp = xtmp.reshape([cfg.nbVarX, cfg.nbData], order='F')
            ftmp, _ = f_reach(cfg, xtmp[:, tl], Mu, Rot)
            cost = ftmp.flatten('F') @ Q @ ftmp.flatten('F') + np.linalg.norm(utmp) * cfg.rfactor

            if cost < cost0 or alpha < 1e-3:
                u = utmp
                print("Iteration {}, cost: {}, alpha: {}".format(i, cost, alpha))
                break

            alpha /= 2

        if np.linalg.norm(alpha * du) < 1e-2:
            break
    return u, x


def uni_ilqr_cp(Mu, Rot, cfg):
    Q, R, idx, tl = get_matrices(cfg)
    PSI, phi = define_control_primitive(cfg)
    Su0, Sx0 = set_dynamical_system(cfg)
    u, x = get_u_x(cfg, Mu, Rot, Q, R, Su0, Sx0, idx, tl, PSI)
    vis(cfg, u, x, Mu, Rot, tl, phi)


def vis(cfg, u, x, Mu, Rot, tl, phi):
    plt.figure()
    plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    # Get points of interest
    f = fkin(cfg, x)
    f00 = fkin0(cfg, x[:, 0])
    f10 = fkin0(cfg, x[:, tl[0]])
    fT0 = fkin0(cfg, x[:, -1])
    u = u.reshape((-1, cfg.nbVarU))

    plt.plot(f00[0, :], f00[1, :], c='black', linewidth=5, alpha=.2)
    plt.plot(f10[0, :], f10[1, :], c='black', linewidth=5, alpha=.4)
    plt.plot(fT0[0, :], fT0[1, :], c='black', linewidth=5, alpha=.6)

    plt.plot(f[0, :], f[1, :], c="black", marker="o", markevery=[0] + tl.tolist())  # ,label="Trajectory"2

    # Plot bounding box or via-points
    ax = plt.gca()
    color_map = ["deepskyblue", "darkorange"]
    for t in range(cfg.nbPoints):

        if cfg.useBoundingBox:
            rect_origin = Mu[:2, t] - Rot[:, :, t] @ np.array(cfg.sz)
            rect_orn = Mu[-1, t]

            rect = patches.Rectangle(rect_origin, cfg.sz[0] * 2, cfg.sz[1] * 2, np.degrees(rect_orn),
                                     color=color_map[t])
            ax.add_patch(rect)
        else:
            plt.scatter(Mu[0, t], Mu[1, t], s=100, marker="X", c=color_map[t])

    # fig, axs = plt.subplots(7, 1)
    #
    # axs[0].plot(f[0, :], c='black')
    # axs[0].set_ylabel("$f(x)_1$")
    # axs[0].set_xticks([0, cfg.nbData])
    # axs[0].set_xticklabels(["0", "T"])
    # for i in range(cfg.nbPoints):
    #     axs[0].scatter(tl[i], Mu[i, 0], c='blue')
    #
    # axs[1].plot(f[1, :], c='black')
    # axs[1].set_ylabel("$f(x)_2$")
    # axs[1].set_xticks([0, cfg.nbData])
    # axs[1].set_xticklabels(["0", "T"])
    # for i in range(cfg.nbPoints):
    #     axs[1].scatter(tl[i], Mu[i, 1], c='blue')
    #
    # axs[2].plot(f[2, :], c='black')
    # axs[2].set_ylabel("$f(x)_3$")
    # axs[2].set_xticks([0, cfg.nbData])
    # axs[2].set_xticklabels(["0", "T"])
    # for i in range(cfg.nbPoints):
    #     axs[2].scatter(tl[i], Mu[i, 2], c='blue')
    #
    # axs[3].plot(u[0, :], c='black')
    # axs[3].set_ylabel("$u_1$")
    # axs[3].set_xticks([0, cfg.nbData - 1])
    # axs[3].set_xticklabels(["0", "T-1"])
    #
    # axs[4].plot(u[1, :], c='black')
    # axs[4].set_ylabel("$u_2$")
    # axs[4].set_xticks([0, cfg.nbData - 1])
    # axs[4].set_xticklabels(["0", "T-1"])
    #
    # axs[5].plot(u[2, :], c='black')
    # axs[5].set_ylabel("$u_3$")
    # axs[5].set_xticks([0, cfg.nbData - 1])
    # axs[5].set_xticklabels(["0", "T-1"])
    #
    # axs[6].set_ylabel("$\phi_k$")
    # axs[6].set_xticks([0, cfg.nbData - 1])
    # axs[6].set_xticklabels(["0", "T-1"])
    # for i in range(cfg.nbFct):
    #     axs[6].plot(phi[:, i])
    # axs[6].set_xlabel("$t$")

    plt.show()


if __name__ == '__main__':
    cfg = get_config('./', 'ilqr')

    # Via-points
    Mu = np.array([[2, 1, -np.pi / 2], [3, 1, -np.pi / 2]]).T  # Via-points
    Rot = np.zeros([2, 2, cfg.nbPoints])  # Object orientation matrices

    # Object rotation matrices
    for t in range(cfg.nbPoints):
        orn_t = Mu[-1, t]
        Rot[:, :, t] = np.asarray([
            [np.cos(orn_t), -np.sin(orn_t)],
            [np.sin(orn_t), np.cos(orn_t)]
        ])

    uni_ilqr_cp(Mu, Rot, cfg)
