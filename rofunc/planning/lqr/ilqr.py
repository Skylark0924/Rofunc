"""
    iLQR for a viapoints task (batch formulation)

    Refers to https://gitlab.idiap.ch/rli/robotics-codes-from-scratch by Dr. Sylvain Calinon
"""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from rofunc.config.get_config import *


def logmap_2d(f, f0):
    position_error = f[:, :2] - f0[:, :2]
    orientation_error = np.imag(np.log(np.exp(f0[:, -1] * 1j).conj().T * np.exp(f[:, -1] * 1j).T)).conj().reshape(
        (-1, 1))
    error = np.hstack([position_error, orientation_error])
    return error


def fkin0(cfg, x):
    L = np.tril(np.ones([cfg.nbVarX, cfg.nbVarX]))
    f = np.vstack([
        L @ np.diag(cfg.l) @ np.cos(L @ x),
        L @ np.diag(cfg.l) @ np.sin(L @ x)
    ]).T
    f = np.vstack([np.zeros(2), f])
    return f


def fk(cfg, x):
    """
    Forward kinematics for end-effector (in robot coordinate system)
    $f(x_t) = x_t$
    """
    # f = x
    x = x.T
    L = np.tril(np.ones([cfg.nbVarX, cfg.nbVarX]))
    f = np.vstack([
        cfg.l @ np.cos(L @ x),
        cfg.l @ np.sin(L @ x),
        np.mod(np.sum(x, 0) + np.pi, 2 * np.pi) - np.pi
    ])  # f1,f2,f3, where f3 is the orientation (single Euler angle for planar robot)
    return f.T


def Jacobian(cfg, x):
    """
    Jacobian with analytical computation (for single time step)
    $J(x_t)= \dfrac{\partial{f(x_t)}}{\partial{x_t}}$
    """
    # J = np.identity(len(x))
    L = np.tril(np.ones([cfg.nbVarX, cfg.nbVarX]))
    J = np.vstack([
        -np.sin(L @ x).T @ np.diag(cfg.l) @ L,
        np.cos(L @ x).T @ np.diag(cfg.l) @ L,
        np.ones(cfg.nbVarX)
    ])
    return J


def f_reach(cfg, robot_state, Mu, Rot, specific_robot=None):
    """
    Error and Jacobian for a via-points reaching task (in object coordinate system)
    Args:
        cfg:
        robot_state: joint state or Cartesian pose
    Returns:

    """
    if specific_robot is not None:
        ee_pose = specific_robot.fk(robot_state)
    else:
        ee_pose = fk(cfg, robot_state)
    f = logmap_2d(ee_pose, Mu)
    J = np.zeros([cfg.nbPoints * cfg.nbVarF, cfg.nbPoints * cfg.nbVarX])
    for t in range(cfg.nbPoints):
        f[t, :2] = Rot[t].T @ f[t, :2]  # Object-oriented forward kinematics
        Jtmp = Jacobian(cfg, robot_state[t])
        Jtmp[:2] = Rot[t].T @ Jtmp[:2]  # Object centered Jacobian

        if cfg.useBoundingBox:
            for i in range(2):
                if abs(f[t, i]) < cfg.sz[i]:
                    f[t, i] = 0
                    Jtmp[i] = 0
                else:
                    f[t, i] -= np.sign(f[t, i]) * cfg.sz[i]

        J[t * cfg.nbVarF:(t + 1) * cfg.nbVarF, t * cfg.nbVarX:(t + 1) * cfg.nbVarX] = Jtmp
    return f, J


def get_matrices(cfg: DictConfig):
    # Precision matrix
    Q = np.identity(cfg.nbVarF * cfg.nbPoints)

    # Control weight matrix
    R = np.identity((cfg.nbData - 1) * cfg.nbVarU) * cfg.rfactor

    # Time occurrence of via-points
    tl = np.linspace(0, cfg.nbData, cfg.nbPoints + 1)
    tl = np.rint(tl[1:]).astype(np.int64) - 1
    idx = np.array([i + np.arange(0, cfg.nbVarX, 1) for i in (tl * cfg.nbVarX)])
    return Q, R, idx, tl


def set_dynamical_system(cfg: DictConfig):
    # Transfer matrices (for linear system as single integrator)
    Su0 = np.vstack([np.zeros([cfg.nbVarX, cfg.nbVarX * (cfg.nbData - 1)]),
                     np.tril(np.kron(np.ones([cfg.nbData - 1, cfg.nbData - 1]), np.eye(cfg.nbVarX) * cfg.dt))])
    Sx0 = np.kron(np.ones(cfg.nbData), np.identity(cfg.nbVarX)).T
    return Su0, Sx0


def get_u_x(cfg: DictConfig, Mu: np.ndarray, Rot: np.ndarray, u: np.ndarray, x0: np.ndarray, Q: np.ndarray,
            R: np.ndarray, Su0: np.ndarray, Sx0: np.ndarray, idx: np.ndarray, tl: np.ndarray):
    Su = Su0[idx.flatten()]  # We remove the lines that are out of interest

    for i in range(cfg.nbIter):
        x = Su0 @ u + Sx0 @ x0  # System evolution
        x = x.reshape([cfg.nbData, cfg.nbVarX])
        f, J = f_reach(cfg, x[tl], Mu, Rot)  # Residuals and Jacobians
        du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + R) @ (
                -Su.T @ J.T @ Q @ f.flatten() - u * cfg.rfactor)  # Gauss-Newton update
        # Estimate step size with backtracking line search method
        alpha = 2
        cost0 = f.flatten() @ Q @ f.flatten() + np.linalg.norm(u) ** 2 * cfg.rfactor  # Cost
        while True:
            utmp = u + du * alpha
            xtmp = Su0 @ utmp + Sx0 @ x0  # System evolution
            xtmp = xtmp.reshape([cfg.nbData, cfg.nbVarX])
            ftmp, _ = f_reach(cfg, xtmp[tl], Mu, Rot)  # Residuals
            cost = ftmp.flatten() @ Q @ ftmp.flatten() + np.linalg.norm(utmp) ** 2 * cfg.rfactor  # Cost
            if cost < cost0 or alpha < 1e-3:
                u = utmp
                print("Iteration {}, cost: {}".format(i, cost))
                break
            alpha /= 2
        if np.linalg.norm(du * alpha) < 1E-2:
            break  # Stop iLQR iterations when solution is reached
    return u, x


def uni(Mu, Rot, u0, x0, cfg, for_test=False):
    Q, R, idx, tl = get_matrices(cfg)
    Su0, Sx0 = set_dynamical_system(cfg)
    u, x = get_u_x(cfg, Mu, Rot, u0, x0, Q, R, Su0, Sx0, idx, tl)
    vis(cfg, Mu, Rot, x, tl, for_test=for_test)


def vis(cfg, Mu, Rot, x, tl, for_test):
    plt.figure()
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')

    # Get points of interest
    f00 = fkin0(cfg, x[0, :])
    f01 = fkin0(cfg, x[tl[0], :])
    f02 = fkin0(cfg, x[tl[1], :])
    f = fk(cfg, x)

    plt.plot(f00[:, 0], f00[:, 1], c='black', linewidth=5, alpha=.2)
    plt.plot(f01[:, 0], f01[:, 1], c='black', linewidth=5, alpha=.4)
    plt.plot(f02[:, 0], f02[:, 1], c='black', linewidth=5, alpha=.6)
    plt.plot(f[:, 0], f[:, 1], c='black', marker='o', markevery=[0] + tl.tolist())

    # Plot bounding box or viapoints
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

    if not for_test:
        plt.show()
