"""
    LQT computed in a recursive way (via-point example)

    Refers to https://gitlab.idiap.ch/rli/robotics-codes-from-scratch by Dr. Sylvain Calinon
"""
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

import rofunc as rf
from rofunc.planning_control.lqt.lqt import LQT
from rofunc.utils.logger.beauty_logger import beauty_print


class LQTFb(LQT):
    def __init__(self, all_points, cfg: DictConfig = None):
        super().__init__(all_points, cfg)

    def get_matrices(self):
        self.cfg.nbPoints = len(self.via_points)

        # Control cost matrix
        R = np.eye(self.cfg.nbVarPos) * self.cfg.rfactor

        # Sparse reference with a set of via-points
        tl = np.linspace(0, self.cfg.nbData - 1, self.cfg.nbPoints + 1)
        tl = np.rint(tl[1:])

        # Definition of augmented precision matrix Qa based on standard precision matrix Q0
        Q0 = np.diag(np.hstack([np.ones(self.cfg.nbVarPos), np.zeros(self.cfg.nbVar - self.cfg.nbVarPos)]))
        Q0_augmented = np.identity(self.cfg.nbVarX)
        Q0_augmented[:self.cfg.nbVar, :self.cfg.nbVar] = Q0

        Q = np.zeros([self.cfg.nbVarX, self.cfg.nbVarX, self.cfg.nbData])
        for i in range(self.cfg.nbPoints):
            Q[:, :, int(tl[i])] = np.vstack([
                np.hstack([np.identity(self.cfg.nbVar), np.zeros([self.cfg.nbVar, 1])]),
                np.hstack([-self.via_points[i, :], 1])]) @ Q0_augmented @ np.vstack([
                np.hstack([np.identity(self.cfg.nbVar), -self.via_points[i, :].reshape([-1, 1])]),
                np.hstack([np.zeros(self.cfg.nbVar), 1])])
        return Q, R, tl

    def set_dynamical_system(self):
        A1d = np.zeros((self.cfg.nbDeriv, self.cfg.nbDeriv))
        for i in range(self.cfg.nbDeriv):
            A1d += np.diag(np.ones((self.cfg.nbDeriv - i,)), i) * self.cfg.dt ** i / np.math.factorial(i)  # Discrete 1D

        B1d = np.zeros((self.cfg.nbDeriv, 1))
        for i in range(0, self.cfg.nbDeriv):
            B1d[self.cfg.nbDeriv - i - 1, :] = self.cfg.dt ** (i + 1) * 1 / np.math.factorial(i + 1)  # Discrete 1D

        A0 = np.kron(A1d, np.eye(self.cfg.nbVarPos))  # Discrete nD
        B0 = np.kron(B1d, np.eye(self.cfg.nbVarPos))  # Discrete nD
        A = np.eye(A0.shape[0] + 1)  # Augmented A
        A[:A0.shape[0], :A0.shape[1]] = A0
        B = np.vstack((B0, np.zeros((1, self.cfg.nbVarPos))))  # Augmented B
        return A, B

    def get_u_x(self, P: np.ndarray, R: np.ndarray, A: np.ndarray, B: np.ndarray, state_noise: np.ndarray):
        x_hat = np.zeros((self.cfg.nbVarX, 2, self.cfg.nbData))
        u_hat = np.zeros((self.cfg.nbVarPos, 2, self.cfg.nbData))
        for n in range(2):
            x = np.hstack([np.zeros(self.cfg.nbVar), 1])
            for t in range(self.cfg.nbData):
                Z_bar = B.T @ P[:, :, t] @ B + R
                K = np.linalg.inv(Z_bar.T @ Z_bar) @ Z_bar.T @ B.T @ P[:, :, t] @ A  # Feedback gain
                u = -K @ x  # Acceleration command with FB on augmented state (resulting in feedback and feedforward terms)
                x = A @ x + B @ u  # Update of state vector

                if t == 25 and n == 1:
                    x += state_noise

                if t == 70 and n == 1:
                    x += state_noise

                if t == 75 and n == 1:
                    x += state_noise

                x_hat[:, n, t] = x  # Log data
                u_hat[:, n, t] = u  # Log data
        return u_hat, x_hat

    def solve(self, state_noise, for_test=False):
        beauty_print('LQT with feedback control', type='module')

        Q, R, tl = self.get_matrices()
        A, B = self.set_dynamical_system()
        P = np.zeros((self.cfg.nbVarX, self.cfg.nbVarX, self.cfg.nbData))
        P[:, :, -1] = Q[:, :, -1]
        for t in range(self.cfg.nbData - 2, -1, -1):
            P[:, :, t] = Q[:, :, t] - A.T @ (
                    P[:, :, t + 1] @ np.dot(B, np.linalg.pinv(B.T @ P[:, :, t + 1] @ B + R))
                    @ B.T @ P[:, :, t + 1] - P[:, :, t + 1]) @ A

        u_hat, x_hat = self.get_u_x(P, R, A, B, state_noise)
        self.vis3d(x_hat, for_test=for_test)
        return u_hat, x_hat

    def vis(self, x_hat, for_test):
        plt.figure()
        for n in range(2):
            plt.plot(x_hat[0, n, :], x_hat[1, n, :], label="Trajectory {}".format(n + 1))
            plt.scatter(x_hat[0, n, 0], x_hat[1, n, 0], marker='o')

        plt.scatter(self.via_points[:, 0], self.via_points[:, 1], s=20 * 1.5 ** 2, marker='o', color="red",
                    label="Via-points")
        plt.legend()
        if not for_test:
            plt.show()

    def vis3d(self, x_hat, for_test):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d', fc='white')

        rf.visualab.traj_plot([x_hat.transpose(1, 2, 0)[0]], mode='3d', ori=False, g_ax=ax, legend='Trajectory 1')
        rf.visualab.traj_plot([x_hat.transpose(1, 2, 0)[1]], mode='3d', ori=False, g_ax=ax, legend='Trajectory 2')

        ax.scatter(self.via_points[:, 0], self.via_points[:, 1], self.via_points[:, 2], s=20 * 1.5 ** 2, marker='o',
                   color="red", label="Via-points")
        plt.legend()
        if not for_test:
            plt.show()
