from typing import Union, List, Tuple

import numpy as np
import pbdlib as pbd
from numpy import ndarray
from pbdlib import HMM

import rofunc as rf
from rofunc.utils.logger.beauty_logger import beauty_print


class TPGMM:
    def __init__(self, demos_x: Union[List, np.ndarray]):
        """
        Task-parameterized Gaussian Mixture Model (TP-GMM)
        :param demos_x: demo displacement
        """
        # TODO: Check list input with different length
        self.demos_x = demos_x

        """
        Some related matrices are generated from the demo data with displacement
        M: Number of demonstrated trajectories in a training set (m will be used as index)
        T: Number of datapoints in a trajectory (t will be used as index)
        P: Number of candidate frames in a task-parameterized mixture (j will be used as index/exponent)
        nb_dim: Dimension of the demo state
        
        demos_xdx: concat original states with their first derivative, [M, T, nb_dim * 2]
        demos_A: the orientation of the p-th candidate coordinate system for this demonstration, [M, T, P, 2, 2]
        demos_b: the position of the p-th candidate coordinate system for this demonstration,  [M, T, P, nb_dim]
        demos_A_xdx: augment demos_A to original states and their first derivative, [M, T, P, nb_dim * 2, nb_dim * 2]
        demos_b_xdx: augment demos_b to original states and their first derivative, [M, T, P, nb_dim * 2]
        demos_xdx_f: states and their first derivative in P frames, [M, T, P, nb_dim * 2]
        demos_xdx_augm: reshape demos_xdx_f, [M, T, nb_dim * 2 * P]
        """
        self.demos_dx, self.demos_A, self.demos_b, self.demos_A_xdx, self.demos_b_xdx, self.demos_xdx, \
            self.demos_xdx_f, self.demos_xdx_augm = self.get_related_matrix()

    def get_dx(self, demos_x=None):
        if demos_x is None:
            demos_x = self.demos_x

        demos_dx = []
        for i in range(len(demos_x)):
            demo_dx = []
            for j in range(len(demos_x[i])):
                if 0 < j < len(demos_x[i]) - 1:
                    dx = (demos_x[i][j + 1] - demos_x[i][j - 1]) / 2
                elif j == len(demos_x[i]) - 1:
                    dx = demos_x[i][j] - demos_x[i][j - 1]
                else:
                    dx = demos_x[i][j + 1] - demos_x[i][j]
                dx = dx / 0.01
                demo_dx.append(dx)
            demos_dx.append(np.array(demo_dx))
        return demos_dx

    def get_A_b(self, demos_x=None):
        if demos_x is None:
            demos_x = self.demos_x

        demos_b = []
        demos_A = []
        for i in range(len(demos_x)):
            demos_b.append(
                np.tile(np.vstack([demos_x[i][0], demos_x[i][-1], ]), (len(demos_x[i]), 1, 1)))
            demos_A.append(np.tile([[[1., 0.], [-0., -1.]], [[1., 0.], [-0., -1.]]], (len(demos_x[i]), 1, 1, 1)))
        demos_A_xdx = [np.kron(np.eye(len(demos_x[0][0])), d) for d in demos_A]
        demos_b_xdx = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b]
        return demos_A, demos_b, demos_A_xdx, demos_b_xdx

    def get_related_matrix(self, demos_x=None):
        if demos_x is None:
            demos_x = self.demos_x

        demos_dx = self.get_dx(demos_x)
        demos_A, demos_b, demos_A_xdx, demos_b_xdx = self.get_A_b(demos_x)

        demos_xdx = [np.hstack([_x, _dx]) for _x, _dx in
                     zip(demos_x, demos_dx)]  # Position and velocity (num_of_points, 14)
        demos_xdx_f = [np.einsum('taji,taj->tai', _A, _x[:, None] - _b) for _x, _A, _b in
                       zip(demos_xdx, demos_A_xdx, demos_b_xdx)]
        demos_xdx_augm = [d.reshape(-1, len(demos_xdx[0][0]) * 2) for d in
                          demos_xdx_f]  # (num_of_points, 28): 0~13 pos-vel in coord 1, 14~27 pos-vel in coord 2
        return demos_dx, demos_A, demos_b, demos_A_xdx, demos_b_xdx, demos_xdx, demos_xdx_f, demos_xdx_augm

    def hmm_learning(self, nb_states: int = 4, reg: float = 1e-3, plot: bool = False) -> pbd.HMM:
        """
        Learn the HMM model by using demos_xdx_augm
        :param nb_states: number of HMM states
        :param reg: [float] or list with [nb_dim x float] for different regularization in different dimensions
                Regularization term used in M-step for covariance matrices
        :param plot: [bool], whether to plot the demo and learned model
        :return: learned HMM model
        """
        beauty_print('{}'.format('learning HMM model'), type='info')

        model = pbd.HMM(nb_states=nb_states)
        model.init_hmm_kbins(self.demos_xdx_augm)  # initializing model
        model.em(self.demos_xdx_augm, reg=reg)

        if plot:
            if int(len(self.demos_xdx_f[0][0, 0]) / 2) == 2:
                rf.tpgmm.hmm_plot(self.demos_xdx_f, model)
            elif int(len(self.demos_xdx_f[0][0, 0]) / 2) > 2:
                rf.tpgmm.hmm_plot_3d(self.demos_xdx_f, model, scale=0.1)
            else:
                raise Exception('Dimension is less than 2, cannot plot')
        return model

    def poe(self, model: pbd.HMM, show_demo_idx: int, task_params: dict = None, plot: bool = False) -> pbd.GMM:
        """
        Product of Expert/Gaussian (PoE), which calculates the mixture distribution from multiple coordinates
        :param model: learned model
        :param show_demo_idx: index of the specific demo to be reproduced
        :param task_params: [dict], task parameters for including transformation matrix A and bias b
        :param plot: [bool], whether to plot the PoE
        :return: The product of experts
        """
        # get transformation for given demonstration.
        if task_params is not None:
            A, b = task_params['A'], task_params['b']
        else:
            # We use the transformation of the first timestep as they are constant
            A, b = self.demos_A_xdx[show_demo_idx][0], self.demos_b_xdx[show_demo_idx][0]
        # transformed model for coordinate system 1
        mod1 = model.marginal_model(slice(0, len(b[0]))).lintrans(A[0], b[0])
        # transformed model for coordinate system 2
        mod2 = model.marginal_model(slice(len(b[0]), len(b[0]) * 2)).lintrans(A[1], b[1])
        # product
        prod = mod1 * mod2

        if plot:
            if len(self.demos_x[0][0]) == 2:
                rf.tpgmm.poe_plot(mod1, mod2, prod, self.demos_x, show_demo_idx)
            elif len(self.demos_x[0][0]) > 2:
                rf.tpgmm.poe_plot_3d(mod1, mod2, prod, self.demos_x, show_demo_idx)
            else:
                raise Exception('Dimension is less than 2, cannot plot')
        return prod

    def _reproduce(self, model: pbd.HMM, prod: pbd.GMM, show_demo_idx: int, plot: bool = False) -> np.ndarray:
        """
        Reproduce the specific demo_idx from the learned model
        :param model: learned model
        :param prod: result of PoE
        :param show_demo_idx: index of the specific demo to be reproduced
        :param plot: [bool], whether to plot the
        :return:
        """
        # get the most probable sequence of state for this demonstration
        sq = model.viterbi(self.demos_xdx_augm[show_demo_idx])

        # solving LQR with Product of Gaussian, see notebook on LQR
        lqr = pbd.PoGLQR(nb_dim=len(self.demos_x[0][0]), dt=0.01, horizon=self.demos_xdx[show_demo_idx].shape[0])
        lqr.mvn_xi = prod.concatenate_gaussian(sq)  # augmented version of gaussian
        lqr.mvn_u = -4
        lqr.x0 = self.demos_xdx[show_demo_idx][0]

        xi = lqr.seq_xi
        if plot:
            if len(self.demos_x[0][0]) == 2:
                rf.tpgmm.generate_plot(xi, prod, self.demos_x, show_demo_idx)
            elif len(self.demos_x[0][0]) > 2:
                rf.tpgmm.generate_plot_3d(xi, prod, self.demos_x, show_demo_idx)
            else:
                raise Exception('Dimension is less than 2, cannot plot')
        return xi

    def fit(self, plot: bool = False) -> pbd.HMM:
        """
        Learning the single arm/agent trajectory representation from demonstration via TP-GMM.
        """
        beauty_print('Learning the trajectory representation from demonstration via TP-GMM')

        model = self.hmm_learning(plot=plot)
        return model

    def reproduce(self, model: pbd.HMM, show_demo_idx: int, plot: bool = False) -> np.ndarray:
        """
        Reproduce the specific demo_idx from the learned model
        """
        beauty_print('reproduce {}-th demo from learned representation'.format(show_demo_idx), type='info')

        prod = self.poe(model, show_demo_idx, plot=plot)
        traj = self._reproduce(model, prod, show_demo_idx, plot=plot)
        return traj

    def generate(self, model: pbd.HMM, ref_demo_idx: int, task_params: dict, plot: bool = False) -> np.ndarray:
        """
        Generate a new trajectory from the learned model
        """
        beauty_print('generate a new demo from learned representation', type='info')

        prod = self.poe(model, ref_demo_idx, task_params, plot=plot)
        traj = self._reproduce(model, prod, ref_demo_idx, plot=plot)
        return traj


class TPGMMBi(TPGMM):
    def __init__(self, demos_left_x: Union[List, np.ndarray], demos_right_x: Union[List, np.ndarray]):
        self.demos_left_x = demos_left_x
        self.demos_right_x = demos_right_x

        self.repr_l = TPGMM(demos_left_x)
        self.repr_r = TPGMM(demos_right_x)

        # self.demos_left_dx, self.demos_left_A, self.demos_left_b, self.demos_left_A_xdx, self.demos_left_b_xdx, \
        #     self.demos_left_xdx, self.demos_left_xdx_f, self.demos_left_xdx_augm = self.get_related_matrix(
        #     self.demos_left_x)
        #
        # self.demos_right_dx, self.demos_right_A, self.demos_right_b, self.demos_right_A_xdx, self.demos_right_b_xdx, \
        #     self.demos_right_xdx, self.demos_right_xdx_f, self.demos_right_xdx_augm = self.get_related_matrix(
        #     self.demos_right_x)

    def fit(self, plot: bool = False) -> Tuple[HMM, HMM]:
        """
        Learning the single arm/agent trajectory representation from demonstration via TP-GMM.
        """
        beauty_print('Learning the trajectory representation from demonstration via TP-GMM')

        model_l = self.repr_l.hmm_learning(plot=plot)
        model_r = self.repr_r.hmm_learning(plot=plot)
        return model_l, model_r

    def reproduce(self, model_l: pbd.HMM, model_r: pbd.HMM, show_demo_idx: int, plot: bool = False) -> Tuple[
        ndarray, ndarray]:
        """
        Reproduce the specific demo_idx from the learned model
        """
        beauty_print('reproduce {}-th demo from learned representation'.format(show_demo_idx), type='info')

        prod_l = self.repr_l.poe(model_l, show_demo_idx, plot=plot)
        prod_r = self.repr_r.poe(model_r, show_demo_idx, plot=plot)
        traj_l = self.repr_l._reproduce(model_l, prod_l, show_demo_idx, plot=plot)
        traj_r = self.repr_r._reproduce(model_r, prod_r, show_demo_idx, plot=plot)

        if plot:
            nb_dim = int(traj_l.shape[1] / 2)
            data_lst = [traj_l[:, :nb_dim], traj_r[:, :nb_dim]]
            rf.visualab.traj_plot(data_lst)
        return traj_l, traj_r

    def generate(self, model_l: pbd.HMM, model_r: pbd.HMM, ref_demo_idx: int, task_params: dict, plot: bool = False) -> \
            Tuple[ndarray, ndarray]:
        """
        Generate a new trajectory from the learned model
        """
        beauty_print('generate a new demo from learned representation', type='info')

        prod_l = self.repr_l.poe(model_l, ref_demo_idx, task_params['Left'], plot=plot)
        prod_r = self.repr_r.poe(model_r, ref_demo_idx, task_params['Right'], plot=plot)
        traj_l = self.repr_l._reproduce(model_l, prod_l, ref_demo_idx, plot=plot)
        traj_r = self.repr_r._reproduce(model_r, prod_r, ref_demo_idx, plot=plot)

        if plot:
            nb_dim = int(traj_l.shape[1] / 2)
            data_lst = [traj_l[:, :nb_dim], traj_r[:, :nb_dim]]
            rf.visualab.traj_plot(data_lst)
        return traj_l, traj_r
