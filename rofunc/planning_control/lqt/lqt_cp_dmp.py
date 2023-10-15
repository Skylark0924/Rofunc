"""
    Linear Quadratic tracker with control primitives applied on a via-point example

    Refers to https://gitlab.idiap.ch/rli/robotics-codes-from-scratch by Dr. Sylvain Calinon
"""

from math import factorial
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from scipy.linalg import block_diag

import rofunc as rf
from rofunc.planning_control.lqt.lqt_cp import LQTCP
from rofunc.utils.logger.beauty_logger import beauty_print


class LQTCPDMP(LQTCP):
    def __init__(self, all_points, cfg: DictConfig = None):
        self.cfg = rf.config.utils.get_config('./planning', 'lqt_cp_dmp') if cfg is None else cfg
        self.all_points = all_points

    def get_matrices(self):
        # Task setting (tracking of acceleration profile and reaching of an end-point)
        Q = np.kron(np.identity(self.cfg.nbData),
                    np.diag(np.concatenate((np.zeros((self.cfg.nbVarU * 2)), np.ones(self.cfg.nbVarU) * 1e-6))))

        Q[-1 - self.cfg.nbVar + 1:-1 - self.cfg.nbVar + 2 * self.cfg.nbVarU + 1,
        -1 - self.cfg.nbVar + 1:-1 - self.cfg.nbVar + 2 * self.cfg.nbVarU + 1] = np.identity(2 * self.cfg.nbVarU) * 1e0

        Mu = self.all_points
        # Weighting matrices in augmented state form
        Qm = np.zeros((self.cfg.nbVarX * self.cfg.nbData, self.cfg.nbVarX * self.cfg.nbData))

        for t in range(self.cfg.nbData):
            id0 = np.linspace(0, self.cfg.nbVar - 1, self.cfg.nbVar, dtype=int) + t * self.cfg.nbVar
            id = np.linspace(0, self.cfg.nbVarX - 1, self.cfg.nbVarX, dtype=int) + t * self.cfg.nbVarX
            Qm[id[0]:id[-1] + 1, id[0]:id[-1] + 1] = np.vstack(
                (np.hstack((np.identity(self.cfg.nbVar), np.zeros((self.cfg.nbVar, 1)))),
                 np.append(-Mu[:, t].reshape(1, -1), 1))) \
                                                     @ block_diag((Q[id0[0]:id0[-1] + 1, id0[0]:id0[-1] + 1]),
                                                                  1) @ np.vstack(
                (np.hstack((np.identity(self.cfg.nbVar), -Mu[:, t].reshape(-1, 1))),
                 np.append(np.zeros((1, self.cfg.nbVar)), 1)))

        Rm = np.identity((self.cfg.nbData - 1) * self.cfg.nbVarU) * self.cfg.rfactor
        return Qm, Rm

    def set_dynamical_system(self):
        A1d = np.zeros(self.cfg.nbDeriv)
        for i in range(self.cfg.nbDeriv):
            A1d = A1d + np.diag(np.ones((1, self.cfg.nbDeriv - i)).flatten(), i) * self.cfg.dt ** i * 1 / factorial(
                i)  # Discrete 1D

        B1d = np.zeros((self.cfg.nbDeriv, 1))
        for i in range(self.cfg.nbDeriv):
            B1d[self.cfg.nbDeriv - 1 - i] = self.cfg.dt ** (i + 1) * 1 / factorial(i + 1)  # Discrete 1D

        A0 = np.kron(A1d, np.eye(self.cfg.nbVarU))  # Discrete nD
        B0 = np.kron(B1d, np.eye(self.cfg.nbVarU))  # Discrete nD

        A = np.vstack((np.hstack((A0, np.zeros((self.cfg.nbVar, 1)))),
                       np.hstack((np.zeros((self.cfg.nbVar)), 1)).reshape(1, -1)))  # Augmented A (homogeneous)
        B = np.vstack((B0, np.zeros((1, self.cfg.nbVarU))))  # Augmented B (homogeneous)

        # Build Sx and Su transfer matrices (for augmented state space)
        Sx = np.kron(np.ones((self.cfg.nbData, 1)), np.eye(self.cfg.nbVarX, self.cfg.nbVarX))
        Su = np.zeros(
            (self.cfg.nbVarX * self.cfg.nbData, self.cfg.nbVarU * (self.cfg.nbData - 1)))  # It's maybe n-1 not sure
        M = B
        for i in range(1, self.cfg.nbData):
            Sx[i * self.cfg.nbVarX:self.cfg.nbData * self.cfg.nbVarX, :] = np.dot(
                Sx[i * self.cfg.nbVarX:self.cfg.nbData * self.cfg.nbVarX, :], A)
            Su[self.cfg.nbVarX * i:self.cfg.nbVarX * i + M.shape[0], 0:M.shape[1]] = M
            M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]
        return Su, Sx, A, B

    def get_u_x(self, mu: np.ndarray, Q: np.ndarray, R: np.ndarray, Su: np.ndarray, Sx: np.ndarray,
                PSI: np.ndarray = None, A: np.ndarray = None, B: np.ndarray = None,
                state_noise: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        # Least squares formulation of recursive LQR with an augmented state space and control primitives
        W = np.linalg.inv(PSI.T @ Su.T @ Q @ Su @ PSI + PSI.T @ R @ PSI) @ PSI.T @ Su.T @ Q @ Sx
        F = PSI @ W  # F with control primitives

        # Reproduction with feedback controller on augmented state space (with CP)
        Ka = np.empty((self.cfg.nbData - 1, self.cfg.nbVarU, self.cfg.nbVarX))
        Ka[0, :, :] = F[0:self.cfg.nbVarU, :]
        P = np.identity(self.cfg.nbVarX)
        for t in range(self.cfg.nbData - 2):
            id = t * self.cfg.nbVarU + np.linspace(2, self.cfg.nbVarU + 1, self.cfg.nbVarU, dtype=int)
            P = P @ np.linalg.pinv(A - B @ Ka[t, :, :])
            Ka[t + 1, :, :] = F[id, :] @ P

        x_hat = np.zeros((2, self.cfg.nbVarX, self.cfg.nbData - 1))
        u_hat = np.zeros((2, self.cfg.nbVarU, self.cfg.nbData - 1))
        for n in range(2):
            x = np.append(mu[:, 0] + np.append(np.array([2, 1]), np.zeros(self.cfg.nbVar - 2)), 1).reshape(-1, 1)
            for t in range(self.cfg.nbData - 1):
                # Feedback control on augmented state (resulting in feedback and feedforward terms on state)
                u = -Ka[t, :, :] @ x
                x = A @ x + B @ u  # Update of state vector
                if t == 24 and n == 1 and state_noise is not None:
                    x = x + state_noise  # Simulated noise on the state
                x_hat[n, :, t] = x.flatten()  # State
        return u_hat, x_hat

    def solve(self, state_noise=None, for_test=False):
        beauty_print('Planning smooth trajectory via LQT (control primitive and DMP)', type='module')

        Q, R = self.get_matrices()
        PSI, phi = self.define_control_primitive()
        Su, Sx, A, B = self.set_dynamical_system()
        mu = self.all_points

        u_hat, x_hat = self.get_u_x(mu, Q, R, Su, Sx, PSI, A, B, state_noise)

        self.vis(x_hat, mu, for_test=for_test)
        return u_hat, x_hat

    def vis(self, x_hat, Mu, for_test, **kwargs):
        plt.figure()
        plt.axis("off")
        plt.gca().set_aspect('equal', adjustable='box')

        plt.plot(Mu[0, :], Mu[1, :], c='blue', linestyle='-', linewidth=2)
        plt.scatter(Mu[0, -1], Mu[1, -1], c='red', s=100)
        plt.scatter(x_hat[0, 0, 0], x_hat[0, 1, 0], c='black', s=50)
        plt.plot(x_hat[0, 0, :], x_hat[0, 1, :], c='black', linestyle=':', linewidth=2)
        plt.plot(x_hat[1, 0, :], x_hat[1, 1, :], c='black', linestyle='-', linewidth=2)
        plt.scatter(x_hat[1, 0, 23:25], x_hat[1, 1, 23:25], c='green', s=30)

        if not for_test:
            plt.show()

    def vis3d(self, data, x_hat):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d', fc='white')

        rf.visualab.traj_plot([x_hat.transpose(0, 2, 1)[0]], mode='3d', ori=False, g_ax=ax, title='Trajectory 1')
        rf.visualab.traj_plot([x_hat.transpose(0, 2, 1)[1]], mode='3d', ori=False, g_ax=ax, title='Trajectory 2')

        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=20 * 1.5 ** 2, marker='o', color="red", label="Via-points")
        plt.legend()
        plt.show()
