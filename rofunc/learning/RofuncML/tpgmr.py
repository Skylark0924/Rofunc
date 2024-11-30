#  Copyright (C) 2024, Junjia Liu
#
#  This file is part of Rofunc.
#
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
#
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
#
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com

from typing import Tuple

import numpy as np
import pbdlib as pbd
from numpy import ndarray
from scipy.linalg import block_diag

import rofunc as rf
from rofunc.learning.RofuncML.tpgmm import TPGMM
from rofunc.utils.logger.beauty_logger import beauty_print


class TPGMR(TPGMM):
    def __init__(self, demos_x, task_params, nb_states: int = 4, reg: float = 1e-3, plot=False):
        """
        Task-parameterized Gaussian Mixture Regression (TP-GMR)
        :param demos_x: demo displacement
        :param task_params: task parameters
        :param nb_states: number of states in the HMM
        :param reg: regularization term
        :param plot: whether to plot the result
        """
        super().__init__(demos_x, task_params, nb_states=nb_states, reg=reg, plot=plot)
        self.gmr = rf.gmr.GMR(self.demos_x, self.demos_dx, self.demos_xdx, nb_states=nb_states, reg=reg,
                              plot=False)

    def gmm_learning(self):
        # Learn the time-dependent GMR from demonstration
        t = np.linspace(0, 10, self.demos_x[0].shape[0])
        demos = [np.hstack([t[:, None], d]) for d in self.demos_xdx_augm]
        self.gmr.demos = demos
        model = self.gmr.gmm_learning()
        mu_gmr, sigma_gmr = self.gmr.estimate(model, t[:, None], dim_in=slice(0, 1),
                                              dim_out=slice(1, 4 * len(self.demos_x[0]) + 1))
        model = pbd.GMM(mu=mu_gmr, sigma=sigma_gmr)
        return model

    def _reproduce(self, model: pbd.HMM, prod: pbd.GMM, show_demo_idx: int, start_xdx: np.ndarray) -> np.ndarray:
        """
        Reproduce the specific demo_idx from the learned model
        :param model: learned model
        :param prod: result of PoE
        :param show_demo_idx: index of the specific demo to be reproduced
        :return:
        """
        lqr = pbd.PoGLQR(nb_dim=len(self.demos_x[0][0]), dt=0.01, horizon=self.demos_xdx[show_demo_idx].shape[0])

        mvn = pbd.MVN()
        mvn.mu = np.concatenate([i for i in prod.mu])
        mvn._sigma = block_diag(*[i for i in prod.sigma])

        lqr.mvn_xi = mvn
        lqr.mvn_u = -4
        lqr.x0 = start_xdx

        xi = lqr.seq_xi
        if self.plot:
            if len(self.demos_x[0][0]) == 2:
                rf.RofuncML.generate_plot(xi, prod, self.demos_x, show_demo_idx)
            elif len(self.demos_x[0][0]) > 2:
                rf.RofuncML.generate_plot_3d(xi, prod, self.demos_x, show_demo_idx, scale=0.1)
            else:
                raise Exception('Dimension is less than 2, cannot plot')
        return xi

    def fit(self):
        beauty_print('Learning the trajectory representation from demonstration via TP-GMR')

        model = self.gmm_learning()
        return model

    def reproduce(self, model, show_demo_idx):
        beauty_print('reproduce {}-th demo from learned representation'.format(show_demo_idx), type='info')

        prod = self.poe(model, show_demo_idx)
        traj = self._reproduce(model, prod, show_demo_idx, self.demos_xdx[show_demo_idx][0])
        return traj, prod

    def generate(self, model: pbd.HMM, ref_demo_idx: int, task_params: dict) -> np.ndarray:
        beauty_print('generate new demo from learned representation', type='info')

        self.get_A_b()

        prod = self.poe(model, ref_demo_idx)
        traj = self._reproduce(model, prod, ref_demo_idx, self.task_params['frame_origins'][0][0])
        return traj, prod


class TPGMRBi(TPGMR):
    def __init__(self, demos_left_x, demos_right_x, task_params, plot=False):
        self.demos_left_x = demos_left_x
        self.demos_right_x = demos_right_x
        self.plot = plot

        self.repr_l = TPGMR(demos_left_x, task_params['left'], plot=plot)
        self.repr_r = TPGMR(demos_right_x, task_params['right'], plot=plot)

    def fit(self):
        beauty_print('Learning the trajectory representation from demonstration via TP-GMR')

        model_l = self.repr_l.gmm_learning()
        model_r = self.repr_r.gmm_learning()
        return model_l, model_r

    def reproduce(self, models, show_demo_idx):
        beauty_print('reproduce {}-th demo from learned representation'.format(show_demo_idx), type='info')

        model_l, model_r = models

        prod_l = self.repr_l.poe(model_l, show_demo_idx)
        prod_r = self.repr_r.poe(model_r, show_demo_idx)
        traj_l = self.repr_l._reproduce(model_l, prod_l, show_demo_idx, self.repr_l.demos_xdx[show_demo_idx][0])
        traj_r = self.repr_r._reproduce(model_r, prod_r, show_demo_idx, self.repr_r.demos_xdx[show_demo_idx][0])

        if self.plot:
            nb_dim = int(traj_l.shape[1] / 2)
            data_lst = [traj_l[:, :nb_dim], traj_r[:, :nb_dim]]
            rf.visualab.traj_plot(data_lst, title='Reproduced bimanual trajectories')
        return traj_l, traj_r, prod_l, prod_r

    def generate(self, models, ref_demo_idx: int, task_params: dict) -> \
            Tuple[ndarray, ndarray]:
        beauty_print('generate new demo from learned representation', type='info')

        model_l, model_r = models

        self.repr_l.task_params = self.task_params['left']
        self.repr_r.task_params = self.task_params['right']

        self.repr_l.get_A_b()
        self.repr_r.get_A_b()

        prod_l = self.repr_l.poe(model_l, ref_demo_idx)
        prod_r = self.repr_r.poe(model_r, ref_demo_idx)
        traj_l = self.repr_l._reproduce(model_l, prod_l, ref_demo_idx, self.repr_l.task_params['frame_origins'][0][0])
        traj_r = self.repr_r._reproduce(model_r, prod_r, ref_demo_idx, self.repr_l.task_params['frame_origins'][0][0])

        if self.plot:
            nb_dim = int(traj_l.shape[1] / 2)
            data_lst = [traj_l[:, :nb_dim], traj_r[:, :nb_dim]]
            rf.visualab.traj_plot(data_lst, title='Generated bimanual trajectories')
        return traj_l, traj_r, prod_l, prod_r
