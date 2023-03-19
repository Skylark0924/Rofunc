import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rofunc.planning_control.lqr.ilqr import get_matrices, set_dynamical_system
from rofunc.config.utils import get_config
from omegaconf import DictConfig


def f_reach(x, Mu):
    f = x - Mu[:, :2]
    J = np.identity(x.shape[1] * x.shape[0])
    return f, J


def f_avoid(cfg, x, Obst, U_obst):
    f = []
    idx = []
    idt = []
    J = np.zeros((0, 0))

    for i in range(x.shape[0]):  # Go through all the via points
        for j in range(Obst.shape[0]):  # Go through all the obstacles
            e = U_obst[j].T @ (x[i] - Obst[j][:2])
            ftmp = 1 - e.T @ e

            if ftmp > 0:
                f.append(ftmp)

                Jtmp = -1 * (U_obst[j] @ e).T.reshape((-1, 1))
                J2 = np.zeros((J.shape[0] + Jtmp.shape[0], J.shape[1] + Jtmp.shape[1]))
                J2[:J.shape[0], :J.shape[1]] = J
                J2[-Jtmp.shape[0]:, -Jtmp.shape[1]:] = Jtmp
                J = J2  # Numpy does not provide a blockdiag function...

                idx.append(i * cfg.nbVarU + np.array(range(cfg.nbVarU)))
                idt.append(i)
    f = np.array(f)
    idx = np.array(idx)
    idt = np.array(idt)
    return f, J.T, idx, idt


def get_u_x(cfg: DictConfig, Mu: np.ndarray, Obst: np.ndarray, U_obst: np.ndarray, u: np.ndarray, x0: np.ndarray,
            R: np.ndarray, Su0: np.ndarray, Sx0: np.ndarray, idx: np.ndarray, tl: np.ndarray):
    Su = Su0[idx.flatten()]  # We remove the lines that are out of interest

    for i in range(cfg.nbIter):
        x = Su0 @ u + Sx0 @ x0
        x = x.reshape((cfg.nbData, cfg.nbVarX))

        f, J = f_reach(x[tl], Mu)  # Tracking objective
        f2, J2, id2, _ = f_avoid(cfg, x, Obst, U_obst)  # Avoidance objective

        if len(id2) > 0:  # Numpy does not allow zero sized array as Indices
            Su2 = Su0[id2.flatten()]
            du = np.linalg.inv(
                Su.T @ J.T @ J @ Su * cfg.Q_track + Su2.T @ J2.T @ J2 @ Su2 * cfg.Q_avoid + R) @ \
                 (-Su.T @ J.T @ f.flatten() * cfg.Q_track - Su2.T @ J2.T @ f2.flatten() * cfg.Q_avoid - u *
                  cfg.rfactor)
        else:  # It means that we have a collision free path
            du = np.linalg.inv(Su.T @ J.T @ J @ Su * cfg.Q_track + R) @ \
                 (-Su.T @ J.T @ f.flatten() * cfg.Q_track - u * cfg.rfactor)

        # Perform line search
        alpha = 1
        cost0 = np.linalg.norm(f.flatten()) ** 2 * cfg.Q_track + np.linalg.norm(
            f2.flatten()) ** 2 * cfg.Q_avoid + np.linalg.norm(u) * cfg.rfactor

        while True:
            utmp = u + du * alpha
            xtmp = Su0 @ utmp + Sx0 @ x0
            xtmp = xtmp.reshape((cfg.nbData, cfg.nbVarX))
            ftmp, _ = f_reach(xtmp[tl], Mu)
            f2tmp, _, _, _ = f_avoid(cfg, xtmp, Obst, U_obst)
            cost = np.linalg.norm(ftmp.flatten()) ** 2 * cfg.Q_track + np.linalg.norm(f2tmp.flatten()) ** 2 * \
                   cfg.Q_avoid + np.linalg.norm(utmp) * cfg.rfactor

            if cost < cost0 or alpha < 1e-3:
                u = utmp
                print("Iteration {}, cost: {}, alpha: {}".format(i, cost, alpha))
                break

            alpha /= 2

        if np.linalg.norm(alpha * du) < 1e-2:  # Early stop condition
            break

    return u, x


def uni_obstacle(Mu, Obst, S_obst, U_obst, u0, x0, cfg, for_test=False):
    Q, R, idx, tl = get_matrices(cfg)
    Su0, Sx0 = set_dynamical_system(cfg)

    u, x = get_u_x(cfg, Mu, Obst, U_obst, u0, x0, R, Su0, Sx0, idx, tl)
    vis(cfg, x, Mu, Obst, S_obst, for_test=for_test)


def vis(cfg, x, Mu, Obst, S_obst, for_test):
    plt.figure()
    plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    plt.scatter(x[0, 0], x[0, 1], c='black', s=100)

    # Plot targets
    for i in range(cfg.nbPoints):
        xt = Mu[i]
        plt.scatter(xt[0], xt[1], c='blue', s=100)

    # Plot obstacles
    al = np.linspace(-np.pi, np.pi, 50)
    ax = plt.gca()
    for i in range(cfg.nbObstacles):
        D, V = np.linalg.eig(S_obst[i])
        D = np.diag(D)
        R = np.real(V @ np.sqrt(D + 0j))
        msh = (R @ np.array([np.cos(al), np.sin(al)])).T + Obst[i][:2]
        p = patches.Polygon(msh, closed=True)
        ax.add_patch(p)

    plt.plot(x[:, 0], x[:, 1], c='black')
    plt.scatter(x[::10, 0], x[::10, 1], c='black')
    if not for_test:
        plt.show()
