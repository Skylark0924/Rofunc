import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from rofunc.config.utils import get_config
from rofunc.planning_control.lqr.ilqr import fk, fkin0, f_reach, set_dynamical_system


def forward_dynamics(cfg, x, u):
    g = 9.81  # Gravity norm
    kv = 1  # Joints Damping
    l = np.reshape(cfg.l, [1, cfg.nbVarX])
    m = np.reshape(cfg.lm, [1, cfg.nbVarX])
    dt = cfg.dt

    nbDOFs = l.shape[1]
    nbData = int(u.shape[0] / nbDOFs + 1)
    Tm = np.multiply(np.triu(np.ones([nbDOFs, nbDOFs])), np.repeat(m, nbDOFs, 0))
    T = np.tril(np.ones([nbDOFs, nbDOFs]))
    Su = np.zeros([2 * nbDOFs * nbData, nbDOFs * (nbData - 1)])

    # Pre-computation of mask (in tensor form)
    S1 = np.zeros([nbDOFs, nbDOFs, nbDOFs])
    J_index = np.ones([1, nbDOFs])
    for j in range(nbDOFs):
        J_index[0, :j] = np.zeros([j])
        S1[:, :, j] = np.repeat(J_index @ np.eye(nbDOFs), nbDOFs, 0) - np.transpose(
            np.repeat(J_index @ np.eye(nbDOFs), nbDOFs, 0))

    # Initialization of dM and dC tensors and A21 matrix
    dM = np.zeros([nbDOFs, nbDOFs, nbDOFs])
    dC = np.zeros([nbDOFs, nbDOFs, nbDOFs])
    A21 = np.zeros([nbDOFs, nbDOFs])

    for t in range(nbData - 1):

        # Computation in matrix form of G,M, and C
        G = -np.reshape(np.sum(Tm, 1), [nbDOFs, 1]) * l.T * np.cos(T @ np.reshape(x[t, 0:nbDOFs], [nbDOFs, 1])) * g
        G = T.T @ G
        M = (l.T * l) * np.cos(np.reshape(T @ x[t, :nbDOFs], [nbDOFs, 1]) - T @ x[t, :nbDOFs]) * (
                Tm ** .5 @ ((Tm ** .5).T))
        M = T.T @ M @ T
        C = -(l.T * l) * np.sin(np.reshape(T @ x[t, :nbDOFs], [nbDOFs, 1]) - T @ x[t, :nbDOFs]) * (
                Tm ** .5 @ ((Tm ** .5).T))

        # Computation in tensor form of derivatives dG,dM, and dC
        dG = np.diagflat(
            np.reshape(np.sum(Tm, 1), [nbDOFs, 1]) * l.T * np.sin(T @ np.reshape(x[t, 0:nbDOFs], [nbDOFs, 1])) * g) @ T
        dM_tmp = (l.T * l) * np.sin(np.reshape(T @ x[t, :nbDOFs], [nbDOFs, 1]) - T @ x[t, :nbDOFs]) * (
                Tm ** .5 @ ((Tm ** .5).T))

        for j in range(dM.shape[2]):
            dM[:, :, j] = T.T @ (dM_tmp * S1[:, :, j]) @ T

        dC_tmp = (l.T * l) * np.cos(np.reshape(T @ x[t, :nbDOFs], [nbDOFs, 1]) - T @ x[t, :nbDOFs]) * (
                Tm ** .5 @ ((Tm ** .5).T))
        for j in range(dC.shape[2]):
            dC[:, :, j] = dC_tmp * S1[:, :, j]

        # update pose
        tau = np.reshape(u[(t) * nbDOFs:(t + 1) * nbDOFs], [nbDOFs, 1])
        inv_M = np.linalg.inv(M)
        ddq = inv_M @ (tau + G + T.T @ C @ (T @ np.reshape(x[t, nbDOFs:], [nbDOFs, 1])) ** 2) - T @ np.reshape(
            x[t, nbDOFs:], [nbDOFs, 1]) * kv

        # compute local linear systems
        x[t + 1, :] = x[t, :] + np.hstack([x[t, nbDOFs:], np.reshape(ddq, [nbDOFs, ])]) * dt
        A11 = np.eye(nbDOFs)
        A12 = A11 * dt
        A22 = np.eye(nbDOFs) + (2 * inv_M @ T.T @ C @ np.diagflat(T @ x[t, nbDOFs:]) @ T - T * kv) * dt
        for j in range(nbDOFs):
            A21[:, j] = (-inv_M @ dM[:, :, j] @ inv_M @ (
                    tau + G + T.T @ C @ (T @ np.reshape(x[t, nbDOFs:], [nbDOFs, 1])) ** 2)
                         + np.reshape(inv_M @ T.T @ dG[:, j], [nbDOFs, 1]) + inv_M @ T.T @ dC[:, :, j] @ (
                                 T @ np.reshape(x[t, nbDOFs:], [nbDOFs, 1])) ** 2).flatten()
        A = np.vstack((np.hstack((A11, A12)), np.hstack((A21 * dt, A22))))
        B = np.vstack((np.zeros([nbDOFs, nbDOFs]), inv_M * dt))

        # compute transformation matrix
        Su[2 * nbDOFs * (t + 1):2 * nbDOFs * (t + 2), :] = A @ Su[2 * nbDOFs * t:2 * nbDOFs * (t + 1), :]
        Su[2 * nbDOFs * (t + 1):2 * nbDOFs * (t + 2), nbDOFs * t:nbDOFs * (t + 1)] = B
    return x, Su


def get_matrices(cfg: DictConfig):
    # Precision matrix
    Q = np.identity(cfg.nbVarF * cfg.nbPoints) * 1e5
    # Control weight matrix
    R = np.identity((cfg.nbData - 1) * cfg.nbVarU) * cfg.rfactor
    # Time occurrence of via-points
    tl = np.linspace(0, cfg.nbData, cfg.nbPoints + 1)
    tl = np.rint(tl[1:]).astype(np.int64) - 1
    idx = np.array([i + np.arange(0, cfg.nbVarX, 1) for i in (tl * 2 * cfg.nbVarX)])
    return Q, R, idx, tl


def get_u_x(cfg: DictConfig, Mu: np.ndarray, Rot: np.ndarray, u: np.ndarray, x0: np.ndarray, v0: np.ndarray,
            Q: np.ndarray, R: np.ndarray, idx: np.ndarray, tl: np.ndarray):
    x = np.zeros([cfg.nbData, 2 * cfg.nbVarX, ])
    x[0, :cfg.nbVarX] = x0
    x[0, cfg.nbVarX:] = v0

    for i in range(cfg.nbIter):
        # system evolution and Transfer matrix (computed from forward dynamics)
        x, Su0 = forward_dynamics(cfg, x, u)
        Su = Su0[idx.flatten()]

        f, J = f_reach(cfg, x[tl, :cfg.nbVarX], Mu, Rot)
        du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + R) @ (-Su.T @ J.T @ Q @ f.flatten() - u * cfg.rfactor)

        # Perform line search
        alpha = 1
        cost0 = f.flatten() @ Q @ f.flatten() + np.linalg.norm(u) * cfg.rfactor

        while True:
            utmp = u + du * alpha
            xtmp, _ = forward_dynamics(cfg, x, utmp)
            ftmp, _ = f_reach(cfg, xtmp[tl, :cfg.nbVarX], Mu, Rot)
            cost = ftmp.flatten() @ Q @ ftmp.flatten() + np.linalg.norm(utmp) * cfg.rfactor

            if cost < cost0 or alpha < 1e-3:
                u = utmp
                print("Iteration {}, cost: {}, alpha: {}".format(i, cost, alpha))
                break
            alpha /= 2
        if abs(cost - cost0) / cost < 1e-3:
            break
    return u, x


def uni_dyna(Mu, Rot, u0, x0, v0, cfg, for_test=False):
    Q, R, idx, tl = get_matrices(cfg)
    Su0, Sx0 = set_dynamical_system(cfg)
    u, x = get_u_x(cfg, Mu, Rot, u0, x0, v0, Q, R, idx, tl)
    vis(cfg, Mu, Rot, x, tl, for_test=for_test)


def vis(cfg, Mu, Rot, x, tl, for_test):
    plt.figure()
    plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    # Get points of interest
    f = fk(cfg, x[:, :cfg.nbVarX])
    f00 = fkin0(cfg, x[0, :cfg.nbVarX])
    fT0 = fkin0(cfg, x[-1, :cfg.nbVarX])

    plt.plot(f00[:, 0], f00[:, 1], c='black', linewidth=5, alpha=.2)
    plt.plot(fT0[:, 0], fT0[:, 1], c='black', linewidth=5, alpha=.6)

    plt.plot(f[:, 0], f[:, 1], c="black", marker="o", markevery=[0] + tl.tolist())  # ,label="Trajectory"

    # Plot bounding box or via-points
    ax = plt.gca()
    color_map = ["deepskyblue", "darkorange"]
    for i in range(cfg.nbPoints):

        if cfg.useBoundingBox:
            rect_origin = Mu[i, :2] - Rot[i] @ np.array(cfg.sz)
            rect_orn = Mu[i, -1]

            rect = patches.Rectangle(rect_origin, cfg.sz[0] * 2, cfg.sz[1] * 2,
                                     np.degrees(rect_orn), color=color_map[i])
            ax.add_patch(rect)
        else:
            plt.scatter(Mu[i, 0], Mu[i, 1], s=100, marker="X", c=color_map[i])

    if not for_test:
        plt.show()
