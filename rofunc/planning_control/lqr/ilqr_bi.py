import numpy as np
import matplotlib.pyplot as plt
from rofunc.planning_control.lqr.ilqr import set_dynamical_system
from rofunc.config.utils import get_config
from omegaconf import DictConfig


def fkin(cfg, x):
    L = np.tril(np.ones(3))
    x = x[np.newaxis, :] if x.ndim == 1 else x
    f = np.vstack((cfg.l[:3] @ np.cos(L @ x[:, :3].T),
                   cfg.l[:3] @ np.sin(L @ x[:, :3].T),
                   np.array(cfg.l)[[0, 3, 4]] @ np.cos(L @ x[:, (0, 3, 4)].T),
                   np.array(cfg.l)[[0, 3, 4]] @ np.sin(L @ x[:, (0, 3, 4)].T)
                   ))
    return f


def fkin0(cfg, x):
    L = np.tril(np.ones(3))
    fl = np.vstack((L @ np.diag(cfg.l[:3]) @ np.cos(L @ x[:3].T),
                    L @ np.diag(cfg.l[:3]) @ np.sin(L @ x[:3].T)
                    ))

    fr = np.vstack((L @ np.diag(np.array(cfg.l)[[0, 3, 4]]) @ np.cos(L @ x[[0, 3, 4]].T),
                    L @ np.diag(np.array(cfg.l)[[0, 3, 4]]) @ np.sin(L @ x[[0, 3, 4]].T)
                    ))

    f = np.hstack((fl[:, ::-1], np.zeros((2, 1)), fr))
    return f


def f_reach(cfg, x, Mu):
    f = fkin(cfg, x) - Mu
    f = f.ravel()
    J = np.zeros([cfg.nbPoints * cfg.nbVarF, cfg.nbPoints * cfg.nbVarX])
    for t in range(x.shape[0]):
        Jtmp = Jkin(cfg, x[t])
        J[t * cfg.nbVarF:(t + 1) * cfg.nbVarF, t * cfg.nbVarX:(t + 1) * cfg.nbVarX] = Jtmp
    return f, J


def Jkin(cfg, x):
    L = np.tril(np.ones(3))
    J = np.zeros((cfg.nbVarF, cfg.nbVarX))
    Ju = np.vstack((-np.sin(L @ x[:3]) @ np.diag(cfg.l[:3]) @ L,
                    np.cos(L @ x[:3]) @ np.diag(cfg.l[:3]) @ L
                    ))

    Jl = np.vstack((-np.sin(L @ x[[0, 3, 4]]) @
                    np.diag(np.array(cfg.l)[[0, 3, 4]]) @ L,
                    np.cos(L @ x[[0, 3, 4]]) @
                    np.diag(np.array(cfg.l)[[0, 3, 4]]) @ L
                    ))
    J[:Ju.shape[0], :Ju.shape[1]] = Ju
    J[2:, (0, 3, 4)] = Jl
    return J


def f_reach_CoM(cfg, x, MuCoM):
    f = fkin_CoM(cfg, x) - np.array([MuCoM]).T
    f = f.ravel(order="F")

    J = np.zeros((2 * x.shape[0], x.shape[0] * cfg.nbVarX))
    for t in range(x.shape[0]):
        Jtmp = Jkin_CoM(cfg, x[t])
        J[t * 2:(t + 1) * 2, t * cfg.nbVarX:(t + 1) * cfg.nbVarX] = Jtmp
    return f, J


def fkin_CoM(cfg, x):
    L = np.tril(np.ones(3))
    f = np.vstack((cfg.l[:3] @ L @ np.cos(L @ x[:, :3].T) +
                   np.array(cfg.l)[[0, 3, 4]] @ L @ np.cos(L @ x[:, (0, 3, 4)].T),
                   cfg.l[:3] @ L @ np.sin(L @ x[:, :3].T) +
                   np.array(cfg.l)[[0, 3, 4]] @ L @ np.sin(L @ x[:, (0, 3, 4)].T)
                   )) / 6
    return f


def Jkin_CoM(cfg, x):
    L = np.tril(np.ones(3))
    Jl = np.vstack((-np.sin(L @ x[:3]) @ L @ np.diag(cfg.l[:3] @ L),
                    np.cos(L @ x[:3]) @ L @ np.diag(cfg.l[:3] @ L)
                    )) / 6
    Jr = np.vstack((-np.sin(L @ x[[0, 3, 4]]) @ L @ np.diag(np.array(cfg.l)[[0, 3, 4]] @ L),
                    np.cos(L @ x[[0, 3, 4]]) @ L @ np.diag(np.array(cfg.l)[[0, 3, 4]] @ L)
                    )) / 6
    J = np.hstack(((Jl[:, 0] + Jr[:, 0])[:, np.newaxis], Jl[:, 1:], Jr[:, 1:]))
    return J


def get_matrices(cfg: DictConfig):
    # Precision matrix
    Q = np.kron(np.eye(cfg.nbPoints), np.diag([1, 1, 0, 0]))
    # Control weight matrix
    R = np.eye(cfg.nbVarU * (cfg.nbData - 1)) * cfg.rfactor
    # Precision matrix for continuous CoM tracking
    Qc = np.kron(np.eye(cfg.nbData), np.diag([1, 1]))

    # Time occurrence of viapoints
    tl = np.linspace(0, cfg.nbData - 1, cfg.nbPoints + 1)
    tl = np.round(tl[1:]).astype(np.int32)
    idx = (tl - 1)[:, np.newaxis] * cfg.nbVarX + np.arange(cfg.nbVarU)
    return Q, Qc, R, idx, tl


def get_u_x(cfg: DictConfig, Mu: np.ndarray, MuCoM: np.ndarray, u: np.ndarray, x0: np.ndarray, Q: np.ndarray,
            Qc: np.ndarray, R: np.ndarray, Su0: np.ndarray, Sx0: np.ndarray, idx: np.ndarray, tl: np.ndarray):
    Su = Su0[idx.flatten()]  # We remove the lines that are out of interest

    for i in range(cfg.nbIter):
        x = Su0 @ u + Sx0 @ x0
        x = x.reshape((cfg.nbData, cfg.nbVarX))
        f, J = f_reach(cfg, x[tl], Mu)  # Forward kinematics and Jacobian for end-effectors
        fc, Jc = f_reach_CoM(cfg, x, MuCoM)  # Forward kinematics and Jacobian for center of mass

        du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + Su0.T @ Jc.T @ Qc @ Jc @ Su0 + R) @ \
             (-Su.T @ J.T @ Q @ f - Su0.T @ Jc.T @ Qc @ fc - u * cfg.rfactor)

        # Estimate step size with line search method
        alpha = 1
        cost0 = f.T @ Q @ f + fc.T @ Qc @ fc + np.linalg.norm(u) ** 2 * cfg.rfactor

        while True:
            utmp = u + du * alpha
            xtmp = (Su0 @ utmp + Sx0 @ x0).reshape((cfg.nbData, cfg.nbVarX))
            ftmp, _ = f_reach(cfg, xtmp[tl], Mu)
            fctmp, _ = f_reach_CoM(cfg, xtmp, MuCoM)

            # for end-effectors and CoM
            cost = ftmp.T @ Q @ ftmp + fctmp.T @ Qc @ fctmp + np.linalg.norm(
                utmp) ** 2 * cfg.rfactor  # for end-effectors and CoM
            if cost < cost0 or alpha < 1e-3:
                print("Iteration {}, cost: {}, alpha: {}".format(i, cost, alpha))
                break  # Stop iLQR when solution is reached

            alpha *= .5

        u = u + du * alpha
        if np.linalg.norm(du * alpha) < 1e-2:
            break
    return u, x


def uni_bi(Mu, MuCoM, u0, x0, cfg, for_test=False):
    Q, Qc, R, idx, tl = get_matrices(cfg)
    Su0, Sx0 = set_dynamical_system(cfg)
    u, x = get_u_x(cfg, Mu, MuCoM, u0, x0, Q, Qc, R, Su0, Sx0, idx, tl)

    vis(cfg, x, Mu, MuCoM, tl, for_test=for_test)


def vis(cfg, x, Mu, MuCoM, tl, for_test):
    tl = np.array([0, tl.item()])

    plt.figure(figsize=(15, 9))
    plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot bimanual robot
    ftmp = fkin0(cfg, x[0])
    plt.plot(ftmp[0], ftmp[1], c='black', linewidth=4, alpha=.2)

    ftmp = fkin0(cfg, x[-1])
    plt.plot(ftmp[0], ftmp[1], c='black', linewidth=4, alpha=.6)

    # Plot CoM
    fc = fkin_CoM(cfg, x)  # Forward kinematics for center of mass
    plt.plot(fc[0, 0], fc[1, 0], c='black', marker='o', linewidth=0,
             markersize=np.sqrt(90), markerfacecolor='none', markeredgewidth=3, alpha=.4)  # Plot CoM
    plt.plot(fc[0, -1], fc[1, -1], c='black', marker='o', linewidth=0,
             markersize=np.sqrt(90), markerfacecolor='none', markeredgewidth=3, alpha=.6)  # Plot CoM

    # Plot end-effectors targets
    for t in range(cfg.nbPoints):
        plt.plot(Mu[0, t], Mu[1, t], marker='o', c='red', markersize=14)

    # Plot CoM target
    plt.plot(MuCoM[0], MuCoM[1], c='red', marker='o', linewidth=0,
             markersize=np.sqrt(90), markerfacecolor='none', markeredgewidth=2, alpha=.8)

    # Plot end-effectors paths
    ftmp = fkin(cfg, x)
    plt.plot(ftmp[0, :], ftmp[1, :], c="black", marker="o", markevery=[0] + tl.tolist())
    plt.plot(ftmp[2, :], ftmp[3, :], c="black", marker="o", markevery=[0] + tl.tolist())
    if not for_test:
        plt.show()
