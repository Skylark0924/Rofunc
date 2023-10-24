"""
    Linear Quadratic Tracker

    Refers to https://gitlab.idiap.ch/rli/robotics-codes-from-scratch by Dr. Sylvain Calinon
"""
from math import factorial
from typing import Tuple

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

import rofunc as rf
from rofunc.utils.logger.beauty_logger import beauty_print


class LQT:
    def __init__(self, all_points, cfg: DictConfig = None):
        self.cfg = rf.config.utils.get_config("./planning", "lqt") if cfg is None else cfg
        self.all_points = all_points
        self.start_point, self.via_points = self._data_process(all_points)

    def _data_process(self, data):
        if len(data[0]) == self.cfg.nbVar:
            all_points = data
        else:
            all_points = np.zeros((len(data), self.cfg.nbVar))
            all_points[:, :self.cfg.nbVarPos] = data
        start_point = all_points[0]
        via_points = all_points[1:]
        return start_point, via_points

    def get_matrices(self):
        self.cfg.nbPoints = len(self.via_points)

        # Control cost matrix
        R = np.identity((self.cfg.nbData - 1) * self.cfg.nbVarPos, dtype=np.float32) * self.cfg.rfactor

        tl = np.linspace(0, self.cfg.nbData, self.cfg.nbPoints + 1)
        tl = np.rint(tl[1:]).astype(np.int64) - 1
        idx_slices = [slice(i, i + self.cfg.nbVar, 1) for i in (tl * self.cfg.nbVar)]

        # Target
        mu = np.zeros((self.cfg.nbVar * self.cfg.nbData, 1), dtype=np.float32)
        # Task precision
        Q = np.zeros((self.cfg.nbVar * self.cfg.nbData, self.cfg.nbVar * self.cfg.nbData), dtype=np.float32)

        for i in range(len(idx_slices)):
            slice_t = idx_slices[i]
            x_t = self.via_points[i].reshape((self.cfg.nbVar, 1))
            mu[slice_t] = x_t

            Q[slice_t, slice_t] = np.diag(
                np.hstack((np.ones(self.cfg.nbVarPos), np.zeros(self.cfg.nbVar - self.cfg.nbVarPos))))
        return mu, Q, R, idx_slices, tl

    def set_dynamical_system(self):
        A1d = np.zeros((self.cfg.nbDeriv, self.cfg.nbDeriv), dtype=np.float32)
        B1d = np.zeros((self.cfg.nbDeriv, 1), dtype=np.float32)
        for i in range(self.cfg.nbDeriv):
            A1d += np.diag(np.ones(self.cfg.nbDeriv - i), i) * self.cfg.dt ** i * 1 / factorial(i)
            B1d[self.cfg.nbDeriv - i - 1] = self.cfg.dt ** (i + 1) * 1 / factorial(i + 1)

        A = np.kron(A1d, np.identity(self.cfg.nbVarPos, dtype=np.float32))
        B = np.kron(B1d, np.identity(self.cfg.nbVarPos, dtype=np.float32))

        # Build Sx and Su transfer matrices
        Su = np.zeros((self.cfg.nbVar * self.cfg.nbData, self.cfg.nbVarPos * (self.cfg.nbData - 1)))
        Sx = np.kron(np.ones((self.cfg.nbData, 1)), np.eye(self.cfg.nbVar, self.cfg.nbVar))

        M = B
        for i in range(1, self.cfg.nbData):
            Sx[i * self.cfg.nbVar:self.cfg.nbData * self.cfg.nbVar, :] = np.dot(
                Sx[i * self.cfg.nbVar:self.cfg.nbData * self.cfg.nbVar, :], A)
            Su[self.cfg.nbVar * i:self.cfg.nbVar * i + M.shape[0], 0:M.shape[1]] = M
            M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]
        return Su, Sx

    def get_u_x(self, mu: np.ndarray, Q: np.ndarray, R: np.ndarray, Su: np.ndarray, Sx: np.ndarray, **kwargs) -> \
            Tuple[np.ndarray, np.ndarray]:
        x0 = self.start_point.reshape((self.cfg.nbVar, 1))
        # Equ. 18
        u_hat = np.linalg.inv(Su.T @ Q @ Su + R) @ Su.T @ Q @ (mu - Sx @ x0)
        # x= S_x x_1 + S_u u
        x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1, self.cfg.nbVar))
        return u_hat, x_hat

    def solve(self):
        beauty_print("Planning smooth trajectory via LQT", type='module')

        mu, Q, R, idx_slices, tl = self.get_matrices()
        Su, Sx = self.set_dynamical_system()
        u_hat, x_hat = self.get_u_x(mu, Q, R, Su, Sx)
        return u_hat, x_hat, mu, idx_slices


class LQTHierarchical(LQT):
    def __init__(self, all_points: np.ndarray, cfg: DictConfig = None, interval: int = 3):
        super().__init__(all_points, cfg)
        self.interval = interval

    def solve(self):
        beauty_print("Planning smooth trajectory via LQT hierarchically", type='module')

        x_hat_lst = []
        for i in tqdm(range(0, len(self.all_points), self.interval)):
            _, self.via_points = self._data_process(self.all_points[i + 1:i + self.interval + 1])

            mu, Q, R, idx_slices, tl = self.get_matrices()
            Su, Sx = self.set_dynamical_system()
            u_hat, x_hat = self.get_u_x(mu, Q, R, Su, Sx)
            self.start_point = x_hat[-1]
            x_hat_lst.append(x_hat)

        x_hat = np.array(x_hat_lst).reshape((-1, self.cfg.nbVar))
        return u_hat, x_hat, mu, idx_slices


class LQTBi(LQT):
    def __init__(self, all_points_l: np.ndarray, all_points_r: np.ndarray, cfg: DictConfig = None):
        self.controller_l = LQT(all_points_l, cfg)
        self.controller_r = LQT(all_points_r, cfg)

    def solve(self):
        beauty_print("Planning smooth bimanual trajectory via LQT", type='module')

        mu_l, Q, R, idx_slices, tl = self.controller_l.get_matrices()
        Su, Sx = self.controller_l.set_dynamical_system()
        u_hat_l, x_hat_l = self.controller_l.get_u_x(mu_l, Q, R, Su, Sx)

        mu_r, Q, R, idx_slices, tl = self.controller_r.get_matrices()
        Su, Sx = self.controller_r.set_dynamical_system()
        u_hat_r, x_hat_r = self.controller_r.get_u_x(mu_r, Q, R, Su, Sx)

        return u_hat_l, u_hat_r, x_hat_l, x_hat_r, mu_l, mu_r, idx_slices
