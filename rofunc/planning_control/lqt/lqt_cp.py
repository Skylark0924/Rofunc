"""
    Linear Quadratic tracker with control primitives applied on a via-point example

    Refers to https://gitlab.idiap.ch/rli/robotics-codes-from-scratch by Dr. Sylvain Calinon
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

import rofunc as rf
from rofunc.planning_control.lqt.lqt import LQT
from rofunc.utils.logger.beauty_logger import beauty_print


class LQTCP(LQT):
    def __init__(self, all_points, cfg: DictConfig = None):
        super().__init__(all_points, cfg)

    def define_control_primitive(self):
        functions = {
            "PIECEWISE": rf.primitive.build_phi_piecewise,
            "RBF": rf.primitive.build_phi_rbf,
            "BERNSTEIN": rf.primitive.build_phi_bernstein,
            "FOURIER": rf.primitive.build_phi_fourier
        }
        phi = functions[self.cfg.basisName](self.cfg.nbData - 1, self.cfg.nbFct)
        PSI = np.kron(phi, np.identity(self.cfg.nbVarPos))
        return PSI, phi

    def set_dynamical_system(self):
        A = np.identity(self.cfg.nbVar)
        if self.cfg.nbDeriv == 2:
            A[:self.cfg.nbVarPos, -self.cfg.nbVarPos:] = np.identity(self.cfg.nbVarPos) * self.cfg.dt

        B = np.zeros((self.cfg.nbVar, self.cfg.nbVarPos))
        derivatives = [self.cfg.dt, self.cfg.dt ** 2 / 2][:self.cfg.nbDeriv]
        for i in range(self.cfg.nbDeriv):
            B[i * self.cfg.nbVarPos:(i + 1) * self.cfg.nbVarPos] = np.identity(self.cfg.nbVarPos) * derivatives[::-1][i]

        # Build Sx and Su transfer matrices
        Su = np.zeros(
            (self.cfg.nbVar * self.cfg.nbData, self.cfg.nbVarPos * (self.cfg.nbData - 1)))  # It's maybe n-1 not sure
        Sx = np.kron(np.ones((self.cfg.nbData, 1)), np.eye(self.cfg.nbVar, self.cfg.nbVar))

        M = B
        for i in range(1, self.cfg.nbData):
            Sx[i * self.cfg.nbVar:self.cfg.nbData * self.cfg.nbVar, :] = np.dot(
                Sx[i * self.cfg.nbVar:self.cfg.nbData * self.cfg.nbVar, :], A)
            Su[self.cfg.nbVar * i:self.cfg.nbVar * i + M.shape[0], 0:M.shape[1]] = M
            M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]
        return Su, Sx

    def get_u_x(self, mu: np.ndarray, Q: np.ndarray, R: np.ndarray, Su: np.ndarray, Sx: np.ndarray,
                PSI: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        x0 = self.start_point.reshape((self.cfg.nbVar, 1))
        w_hat = np.linalg.inv(PSI.T @ Su.T @ Q @ Su @ PSI + PSI.T @ R @ PSI) @ PSI.T @ Su.T @ Q @ (mu - Sx @ x0)
        u_hat = PSI @ w_hat
        x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1, self.cfg.nbVar))
        u_hat = u_hat.reshape((-1, self.cfg.nbVarPos))
        return u_hat, x_hat

    def solve(self):
        beauty_print("Planning smooth trajectory via LQT (control primitive)", type='module')

        mu, Q, R, idx_slices, tl = self.get_matrices()
        PSI, phi = self.define_control_primitive()
        Su, Sx = self.set_dynamical_system()
        u_hat, x_hat = self.get_u_x(mu, Q, R, Su, Sx, PSI)

        # self.vis(param, x_hat, u_hat, muQ, idx_slices, tl, phi)
        # rf.visualab.traj_plot([x_hat[:, :2]])
        return u_hat, x_hat, mu, idx_slices

    def vis(self, x_hat, u_hat, muQ, idx_slices, tl, phi):
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
        axs[0].set_xticks([0, self.cfg.nbData])
        axs[0].set_xticklabels(["0", "T"])
        for t in tl:
            axs[0].scatter(t, x_hat[t, 0], c='blue')

        axs[1].plot(x_hat[:, 1], c='black')
        axs[1].set_ylabel("$x_2$")
        axs[1].set_xticks([0, self.cfg.nbData])
        axs[1].set_xticklabels(["0", "T"])
        for t in tl:
            axs[1].scatter(t, x_hat[t, 1], c='blue')

        axs[2].plot(u_hat[:, 0], c='black')
        axs[2].set_ylabel("$u_1$")
        axs[2].set_xticks([0, self.cfg.nbData - 1])
        axs[2].set_xticklabels(["0", "T-1"])

        axs[3].plot(u_hat[:, 1], c='black')
        axs[3].set_ylabel("$u_2$")
        axs[3].set_xticks([0, self.cfg.nbData - 1])
        axs[3].set_xticklabels(["0", "T-1"])

        axs[4].set_ylabel("$\phi_k$")
        axs[4].set_xticks([0, self.cfg.nbData - 1])
        axs[4].set_xticklabels(["0", "T-1"])
        for i in range(self.cfg.nbFct):
            axs[4].plot(phi[:, i])
        axs[4].set_xlabel("$t$")

        plt.show()
