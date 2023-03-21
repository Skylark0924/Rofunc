import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from omegaconf import DictConfig

from rofunc.config.utils import get_config
from rofunc.planning_control.lqr.ilqr import get_matrices, set_dynamical_system


def fkin0(cfg, x):
    T = np.tril(np.ones([cfg.nbVarX, cfg.nbVarX]))
    T2 = np.tril(np.matlib.repmat(cfg.l, len(x), 1))
    f = np.vstack((
        T2 @ np.cos(T @ x),
        T2 @ np.sin(T @ x)
    )).T
    f = np.vstack((np.zeros(2), f))
    return f


def fk(cfg, x):
    x = x.T
    A = np.tril(np.ones([cfg.nbVarX, cfg.nbVarX]))
    f = np.vstack((cfg.l @ np.cos(A @ x),
                   cfg.l @ np.sin(A @ x)))
    return f


def jkin(cfg, x):
    T = np.tril(np.ones((len(x), len(x))))
    J = np.vstack((
        -np.sin(T @ x).T @ np.diag(cfg.l) @ T,
        np.cos(T @ x).T @ np.diag(cfg.l) @ T
    ))
    return J


def f_reach(cfg, x, Mu):
    f = fk(cfg, x).T - Mu
    J = np.zeros((len(x) * cfg.nbVarF, len(x) * cfg.nbVarX))

    for t in range(x.shape[0]):
        Jtmp = jkin(cfg, x[t])
        J[t * cfg.nbVarF:(t + 1) * cfg.nbVarF, t * cfg.nbVarX:(t + 1) * cfg.nbVarX] = Jtmp
    return f.T, J


def fkin_CoM(cfg, x):
    x = x.T
    A = np.tril(np.ones([cfg.nbVarX, cfg.nbVarX]))
    f = np.vstack((cfg.l @ A @ np.cos(A @ x),
                   cfg.l @ A @ np.sin(A @ x))) / cfg.nbVarX
    return f


def f_reach_CoM(cfg, x, MuCoM):
    f = fkin_CoM(cfg, x).T - MuCoM
    J = np.zeros((len(x) * cfg.nbVarF, len(x) * cfg.nbVarX))
    A = np.tril(np.ones([cfg.nbVarX, cfg.nbVarX]))
    for t in range(x.shape[0]):
        Jtmp = np.vstack((-np.sin(A @ x[t]).T @ A @ np.diag(cfg.l @ A),
                          np.cos(A @ x[t]).T @ A @ np.diag(cfg.l @ A))) / cfg.nbVarX
        if cfg.useBoundingBox:
            for i in range(1):
                if abs(f[t, i]) < cfg.szCoM:
                    f[t, i] = 0
                    Jtmp[i] = 0
                else:
                    f[t, i] -= np.sign(f[t, i]) * cfg.szCoM
        J[t * cfg.nbVarF:(t + 1) * cfg.nbVarF, t * cfg.nbVarX:(t + 1) * cfg.nbVarX] = Jtmp
    f = f.flatten().T
    return f, J


def get_u_x(cfg: DictConfig, Mu: np.ndarray, MuCoM: np.ndarray, u: np.ndarray, x0: np.ndarray, Q: np.ndarray,
            Qc: np.ndarray, R: np.ndarray, Su0: np.ndarray, Sx0: np.ndarray, idx: np.ndarray, tl: np.ndarray):
    Su = Su0[idx.flatten()]  # We remove the lines that are out of interest

    for i in range(cfg.nbIter):
        x = Su0 @ u + Sx0 @ x0
        x = x.reshape((cfg.nbData, cfg.nbVarX))

        f, J = f_reach(cfg, x[tl], Mu)
        fc, Jc = f_reach_CoM(cfg, x, MuCoM)
        du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + Su0.T @ Jc.T @ Qc @ Jc @ Su0 + R) @ (
                -Su.T @ J.T @ Q @ f.flatten() - Su0.T @ Jc.T @ Qc @ fc.flatten() - u * cfg.rfactor)

        # Perform line search
        alpha = 1
        cost0 = f.flatten() @ Q @ f.flatten() + fc.flatten() @ Qc @ fc.flatten() + np.linalg.norm(u) * cfg.rfactor

        while True:
            utmp = u + du * alpha
            xtmp = Su0 @ utmp + Sx0 @ x0
            xtmp = xtmp.reshape((cfg.nbData, cfg.nbVarX))
            ftmp, _ = f_reach(cfg, xtmp[tl], Mu)
            fctmp, _ = f_reach_CoM(cfg, xtmp, MuCoM)
            cost = ftmp.flatten() @ Q @ ftmp.flatten() + fctmp.T @ Qc @ fctmp + np.linalg.norm(utmp) * cfg.rfactor

            if cost < cost0 or alpha < 1e-3:
                u = utmp
                print("Iteration {}, cost: {}, alpha: {}".format(i, cost, alpha))
                break

            alpha /= 2
    return u, x


def uni_com(Mu, MuCoM, u0, x0, cfg, for_test=False):
    Q, R, idx, tl = get_matrices(cfg)
    Qc = np.kron(np.identity(cfg.nbData), np.diag([1E0, 0]))
    Su0, Sx0 = set_dynamical_system(cfg)
    u, x = get_u_x(cfg, Mu, MuCoM, u0, x0, Q, Qc, R, Su0, Sx0, idx, tl)

    vis(cfg, x, Mu, MuCoM, for_test=for_test)


def vis(cfg, x, Mu, MuCoM, for_test):
    plt.figure()
    plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    # plot ground
    plt.plot([-1, 3], [0, 0], linestyle='-', c=[.2, .2, .2], linewidth=2)

    # Get points of interest
    f00 = fkin0(cfg, x[0])
    fT0 = fkin0(cfg, x[-1])
    fc = fkin_CoM(cfg, x)

    plt.plot(f00[:, 0], f00[:, 1], c=[.8, .8, .8], linewidth=4, linestyle='-')
    plt.plot(fT0[:, 0], fT0[:, 1], c=[.4, .4, .4], linewidth=4, linestyle='-')

    # plot CoM
    plt.plot(fc[0, 0], fc[1, 0], c=[.5, .5, .5], marker="o", markeredgewidth=4, markersize=8, markerfacecolor='white')
    plt.plot(fc[0, -1], fc[1, -1], c=[.2, .2, .2], marker="o", markeredgewidth=4, markersize=8, markerfacecolor='white')

    # plot end-effector target
    plt.plot(Mu[0], Mu[1], marker="o", markersize=8, c="r")

    # Plot bounding box or via-points
    ax = plt.gca()
    for i in range(cfg.nbPoints):
        if cfg.useBoundingBox:
            rect_origin = MuCoM + np.array([0, 3.5]) - np.array([cfg.szCoM, 3.5])
            rect = patches.Rectangle(rect_origin, cfg.szCoM * 2, 3.5 * 2,
                                     facecolor=[.8, 0, 0], alpha=0.1, edgecolor=None)
            ax.add_patch(rect)
    if not for_test:
        plt.show()