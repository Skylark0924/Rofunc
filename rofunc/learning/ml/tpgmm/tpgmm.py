from typing import Union, List, Tuple

import math
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

import rofunc as rf
from rofunc.utils.logger.beauty_logger import beauty_print
from rofunc.learning.ml.gmm.gmm import GMM
from rofunc.learning.ml.hmm.hmm import HMM


class TPGMM:
    def __init__(self, demos_x: Union[List, np.ndarray], nb_states: int = 4, reg: float = 1e-3, horizon: int = 150,
                 P: int = 2, plot: bool = False, task_params: dict = None):
        """
        Task-parameterized Gaussian Mixture Model (TP-GMM)
        :param demos_x: demo displacement
        :param nb_states: number of states in the HMM
        :param reg: regularization coefficient
        :param horizon: horizon of the reproduced trajectory
        :param P: number of candidate frames in a task-parameterized mixture
        :param plot: whether to plot the result
        :param task_params: task parameters
        """
        self.demos_x = demos_x
        self.nb_states = nb_states
        self.reg = reg
        self.horizon = horizon
        self.plot = plot
        self.nb_dim = len(self.demos_x[0][0])
        self.nb_deriv = 2  # TODO: for now it is just original state and its first derivative
        self.P = P

        """
        Some related matrices are generated from the demo data with displacement
        M: Number of demonstrated trajectories in a training set (m will be used as index)
        T: Number of datapoints in a trajectory (t will be used as index)
        P: Number of candidate frames in a task-parameterized mixture (j will be used as index/exponent)
        nb_dim: Dimension of the demo state
        nb_deriv: Number of derivatives of the demo state
        
        demos_xdx: concat original states with their first derivative, [M, T, nb_dim * nb_deriv]
        demos_A: the orientation of the p-th candidate coordinate system for this demonstration, [M, T, P, nb_dim, nb_dim]
        demos_b: the position of the p-th candidate coordinate system for this demonstration,  [M, T, P, nb_dim]
        demos_A_xdx: augment demos_A to original states and their first derivative, [M, T, P, nb_dim * nb_deriv, nb_dim * nb_deriv]
        demos_b_xdx: augment demos_b to original states and their first derivative, [M, T, P, nb_dim * nb_deriv]
        demos_xdx_f: states and their first derivative in P frames, [M, T, P, nb_dim * nb_deriv]
        demos_xdx_augm: reshape demos_xdx_f, [M, T, nb_dim * nb_deriv * P]
        """
        self.demos_dx, self.demos_A, self.demos_b, self.demos_A_xdx, self.demos_b_xdx, self.demos_xdx, \
            self.demos_xdx_f, self.demos_xdx_augm = self.get_related_matrix(task_params)

    def get_dx(self):
        demos_dx = []
        for i in range(len(self.demos_x)):
            demo_dx = []
            for j in range(len(self.demos_x[i])):
                if 0 < j < len(self.demos_x[i]) - 1:
                    dx = (self.demos_x[i][j + 1] - self.demos_x[i][j - 1]) / 2
                elif j == len(self.demos_x[i]) - 1:
                    dx = self.demos_x[i][j] - self.demos_x[i][j - 1]
                else:
                    dx = self.demos_x[i][j + 1] - self.demos_x[i][j]
                dx = dx / 0.01
                demo_dx.append(dx)
            demos_dx.append(np.array(demo_dx))
        return demos_dx

    def get_A_b(self):
        """
        A general task parameter setting: Set the start and the end point as two candidate frames
        :return: demos_A, demos_b, demos_A_xdx, demos_b_xdx
        """
        demos_b = []
        demos_A = []
        for i in range(len(self.demos_x)):
            demos_b.append(np.tile(np.vstack([self.demos_x[i][0], self.demos_x[i][-1]]), (len(self.demos_x[i]), 1, 1)))
            demos_A.append(np.tile([np.eye(self.nb_dim)] * self.P, (len(self.demos_x[i]), 1, 1, 1)))
        demos_A_xdx = [np.kron(np.eye(self.nb_deriv), demos_A[i]) for i in range(len(demos_A))]
        demos_b_xdx = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b]
        return demos_A, demos_b, demos_A_xdx, demos_b_xdx

    def get_A_b_from_task_params(self, task_params: dict):
        """
        Define custom task parameters
        :param task_params: task parameters
        :param show_demo_idx: the index of the demo to show
        :return: demos_A, demos_b, demos_A_xdx, demos_b_xdx
        """
        if 'start_xdx' in task_params and 'end_xdx' in task_params:
            # TODO: only for one demo now
            start_xdx, end_xdx = task_params['start_xdx'], task_params['end_xdx']
            demos_b = np.tile(np.vstack([start_xdx[: self.nb_dim], end_xdx[: self.nb_dim]]),
                              (self.horizon, 1, 1))
            demos_A = np.tile([np.eye(self.nb_dim)] * self.P, (self.horizon, 1, 1, 1))

            demos_b_xdx = np.concatenate([demos_b, np.zeros(demos_b.shape)], axis=-1)
            demos_A_xdx = np.kron(np.eye(self.nb_deriv), demos_A)
            task_params_A_b = {'A': demos_A_xdx[0], 'b': demos_b_xdx[0]}
        elif 'demos_A_xdx' in task_params and 'demos_b_xdx' in task_params:
            demos_A, demos_b, demos_A_xdx, demos_b_xdx = task_params['demos_A'], task_params['demos_b'], task_params[
                'demos_A_xdx'], task_params['demos_b_xdx']
            task_params_A_b = task_params
        else:
            raise ValueError('Invalid task_params')

        return demos_A, demos_b, demos_A_xdx, demos_b_xdx, task_params_A_b

    def get_related_matrix(self, task_params: dict = None):
        demos_dx = self.get_dx()
        if task_params is None:
            demos_A, demos_b, demos_A_xdx, demos_b_xdx = self.get_A_b()
        else:
            demos_A, demos_b, demos_A_xdx, demos_b_xdx, _ = self.get_A_b_from_task_params(task_params)

        demos_xdx = [np.hstack([_x, _dx]) for _x, _dx in
                     zip(self.demos_x, demos_dx)]  # Position and velocity (num_of_points, 14)
        demos_xdx_f = [np.einsum('taji,taj->tai', _A, _x[:, None] - _b) for _x, _A, _b in
                       zip(demos_xdx, demos_A_xdx, demos_b_xdx)]
        demos_xdx_augm = [d.reshape(-1, self.nb_deriv * self.P * self.nb_dim) for d in
                          demos_xdx_f]  # (num_of_points, 28): 0~13 pos-vel in coord 1, 14~27 pos-vel in coord 2
        return demos_dx, demos_A, demos_b, demos_A_xdx, demos_b_xdx, demos_xdx, demos_xdx_f, demos_xdx_augm

    def hmm_learning(self) -> HMM:
        beauty_print('{}'.format('learning HMM model'), type='info')

        model = HMM(nb_states=self.nb_states)
        model.init_hmm_kbins(self.demos_xdx_augm)  # initializing model
        model.em(self.demos_xdx_augm, reg=self.reg)

        if self.plot:
            if int(len(self.demos_xdx_f[0][0, 0]) / 2) == 2:
                rf.tpgmm.hmm_plot(self.demos_xdx_f, model)
            elif int(len(self.demos_xdx_f[0][0, 0]) / 2) > 2:
                rf.tpgmm.hmm_plot_3d(self.demos_xdx_f, model, scale=0.1)
            else:
                raise Exception('Dimension is less than 2, cannot plot')
        return model

    def poe(self, model: HMM, show_demo_idx: int, task_params: dict = None) -> GMM:
        """
        Product of Expert/Gaussian (PoE), which calculates the mixture distribution from multiple coordinates
        :param model: learned model
        :param show_demo_idx: index of the specific demo to be reproduced
        :param task_params: [dict], task parameters for including transformation matrix A and bias b
        :return: The product of experts
        """
        mod_list = []

        # get transformation for given demonstration.
        if task_params is not None:
            A, b = task_params['A'], task_params['b']
            for p in range(self.P):
                # transformed model for coordinate system p
                mod_list.append(model.marginal_model(slice(p * len(b[0]), (p + 1) * len(b[0]))).lintrans(A[p], b[p]))
        else:
            # We use the transformation of the first timestep as they are constant
            A, b = self.demos_A_xdx[show_demo_idx][::math.ceil(len(self.demos_A_xdx[show_demo_idx]) / 4)], \
                self.demos_b_xdx[show_demo_idx][::math.ceil(len(self.demos_A_xdx[show_demo_idx]) / 4)]
            for p in range(self.P):
                # transformed model for coordinate system p
                # mod_list.append(model.marginal_model(slice(p * len(b[0]), (p + 1) * len(b[0]))).lintrans(A[p], b[p]))
                mod_list.append(
                    model.marginal_model(
                        slice(p * self.nb_deriv * self.nb_dim, (p + 1) * self.nb_deriv * self.nb_dim)).lintrans_dyna(
                        A[:, p], b[:, p]))

        # product
        prod = mod_list[0]
        for p in range(1, self.P):
            prod *= mod_list[p]

        if self.plot:
            if self.nb_dim == 2:
                rf.tpgmm.poe_plot(mod_list[0], mod_list[1], prod, self.demos_x, show_demo_idx)
            elif self.nb_dim > 2:
                rf.tpgmm.poe_plot_3d(mod_list[0], mod_list[1], prod, self.demos_x, show_demo_idx)
            else:
                raise Exception('Dimension is less than 2, cannot plot')
        return prod

    def _reproduce(self, model: HMM, prod: GMM, show_demo_idx: int,
                   start_xdx: np.ndarray) -> np.ndarray:
        """
        Reproduce the specific demo_idx from the learned model
        :param model: learned model
        :param prod: result of PoE
        :param show_demo_idx: index of the specific demo to be reproduced
        :param start_xdx: start state
        :return:
        """
        # get the most probable sequence of state for this demonstration
        sq = model.viterbi(self.demos_xdx_augm[show_demo_idx])

        # solving LQR with Product of Gaussian, see notebook on LQR
        lqr = rf.lqr.PoGLQR(nb_dim=self.nb_dim, dt=0.01, horizon=self.demos_xdx[show_demo_idx].shape[0])
        lqr.mvn_xi = prod.concatenate_gaussian(sq)  # augmented version of gaussian
        lqr.mvn_u = -4
        lqr.x0 = start_xdx

        xi = lqr.seq_xi
        if self.plot:
            if self.nb_dim == 2:
                rf.tpgmm.generate_plot(xi, prod, self.demos_x, show_demo_idx)
            elif self.nb_dim > 2:
                rf.tpgmm.generate_plot_3d(xi, prod, self.demos_x, show_demo_idx)
            else:
                raise Exception('Dimension is less than 2, cannot plot')
        return xi

    def fit(self) -> HMM:
        """
        Learning the single arm/agent trajectory representation from demonstration via TP-GMM.
        """
        beauty_print('Learning the trajectory representation from demonstration via TP-GMM')

        model = self.hmm_learning()
        return model

    def reproduce(self, model: HMM, show_demo_idx: int) -> Tuple[np.ndarray, GMM]:
        """
        Reproduce the specific demo_idx from the learned model
        """
        beauty_print('reproduce {}-th demo from learned representation'.format(show_demo_idx), type='info')

        prod = self.poe(model, show_demo_idx)
        traj = self._reproduce(model, prod, show_demo_idx, self.demos_xdx[show_demo_idx][0])
        return traj, prod

    def generate(self, model: HMM, ref_demo_idx: int, task_params: dict) -> Tuple[np.ndarray, GMM]:
        """
        Generate a new trajectory from the learned model
        """
        beauty_print('generate a new demo from learned representation', type='info')

        _, _, _, _, task_params_A_b = self.get_A_b_from_task_params(task_params)

        prod = self.poe(model, ref_demo_idx, task_params_A_b)
        traj = self._reproduce(model, prod, ref_demo_idx, task_params['start_xdx'])
        return traj, prod


class TPGMMBi(TPGMM):
    """
    Simple TPGMMBi (no coordination)
    """

    def __init__(self, demos_left_x: Union[List, np.ndarray], demos_right_x: Union[List, np.ndarray],
                 nb_states: int = 4, reg: float = 1e-3, horizon: int = 150, plot: bool = False):
        assert len(demos_left_x) == len(
            demos_right_x), 'The number of demonstrations for left and right arm should be the same'
        assert len(demos_left_x[0]) == len(
            demos_right_x[0]), 'The number of timesteps for left and right arm should be the same'
        self.demos_left_x = demos_left_x
        self.demos_right_x = demos_right_x
        self.plot = plot

        self.repr_l = TPGMM(demos_left_x, nb_states, reg, horizon, plot)
        self.repr_r = TPGMM(demos_right_x, nb_states, reg, horizon, plot)

    def fit(self) -> Tuple[HMM, HMM]:
        """
        Learning the single arm/agent trajectory representation from demonstration via TP-GMM.
        """
        beauty_print('Learning the bimanual trajectory representation from demonstration via TP-GMM')

        model_l = self.repr_l.hmm_learning()
        model_r = self.repr_r.hmm_learning()
        return model_l, model_r

    def reproduce(self, model_l: HMM, model_r: HMM, show_demo_idx: int) -> Tuple[
        ndarray, ndarray, GMM, GMM]:
        """
        Reproduce the specific demo_idx from the learned model
        """
        beauty_print('reproduce {}-th demo from learned representation'.format(show_demo_idx), type='info')

        prod_l = self.repr_l.poe(model_l, show_demo_idx)
        prod_r = self.repr_r.poe(model_r, show_demo_idx)
        traj_l = self.repr_l._reproduce(model_l, prod_l, show_demo_idx, self.repr_l.demos_xdx[show_demo_idx][0])
        traj_r = self.repr_r._reproduce(model_r, prod_r, show_demo_idx, self.repr_r.demos_xdx[show_demo_idx][0])

        if self.plot:
            data_lst = [traj_l[:, :self.repr_l.nb_dim], traj_r[:, :self.repr_r.nb_dim]]
            rf.visualab.traj_plot(data_lst)
        return traj_l, traj_r, prod_l, prod_r

    def generate(self, model_l: HMM, model_r: HMM, ref_demo_idx: int, task_params: dict) -> \
            Tuple[ndarray, ndarray, GMM, GMM]:
        """
        Generate a new trajectory from the learned model
        """
        beauty_print('generate a new demo from learned representation', type='info')

        _, _, _, _, task_params_A_b_l = self.repr_l.get_A_b_from_task_params(task_params['Left'])
        _, _, _, _, task_params_A_b_r = self.repr_r.get_A_b_from_task_params(task_params['Right'])

        prod_l = self.repr_l.poe(model_l, ref_demo_idx, task_params_A_b_l)
        prod_r = self.repr_r.poe(model_r, ref_demo_idx, task_params_A_b_r)
        traj_l = self.repr_l._reproduce(model_l, prod_l, ref_demo_idx, task_params['Left']['start_xdx'])
        traj_r = self.repr_r._reproduce(model_r, prod_r, ref_demo_idx, task_params['Right']['start_xdx'])

        if self.plot:
            data_lst = [traj_l[:, :self.repr_l.nb_dim], traj_r[:, :self.repr_r.nb_dim]]
            rf.visualab.traj_plot(data_lst)
        return traj_l, traj_r, prod_l, prod_r


class TPGMMBiLQRBiCoord(TPGMMBi):
    """
    Simple TPGMMBi (no coordination) with bimanual coordination in the LQR controller
    """

    def __init__(self, demos_left_x, demos_right_x, nb_states: int = 4, reg: float = 1e-3, horizon: int = 150,
                 plot: bool = False):
        super().__init__(demos_left_x, demos_right_x, nb_states, reg, horizon, plot)

        self.demos_com_x = self.get_rel_demos()
        self.repr_c = rf.tpgmm.TPGMM(self.demos_com_x, nb_states, reg, horizon, plot)

    def get_rel_demos(self):
        # calculate the relative movement of each demo
        rel_demos = np.zeros_like(self.demos_left_x)
        if self.plot:
            plt.figure()
            for i in range(self.demos_left_x.shape[0]):
                for j in range(self.demos_left_x.shape[1]):
                    # rel_demos[i, j] = np.linalg.norm(left_x[i, j] - right_x[i, j])
                    rel_demos[i, j] = self.demos_left_x[i, j] - self.demos_right_x[i, j]

                plt.plot(rel_demos[i, :, 0], rel_demos[i, :, 1])
            plt.axis('equal')
            plt.legend()
            plt.show()
        return rel_demos

    def _bi_reproduce(self, model_l, prod_l, model_r, prod_r, model_c, prod_c, show_demo_idx):
        # get the most probable sequence of state for this demonstration
        sq_l = model_l.viterbi(self.repr_l.demos_xdx_augm[show_demo_idx])
        sq_r = model_r.viterbi(self.repr_r.demos_xdx_augm[show_demo_idx])
        sq_c = model_c.viterbi(self.repr_c.demos_xdx_augm[show_demo_idx])

        # solving LQR with Product of Gaussian, see notebook on LQR
        lqr = rf.lqr.PoGLQRBi(nb_dim=self.nb_dim, dt=0.01, horizon=self.demos_left_x[show_demo_idx].shape[0])
        lqr.mvn_xi_l = prod_l.concatenate_gaussian(sq_l)  # augmented version of gaussian
        lqr.mvn_xi_r = prod_r.concatenate_gaussian(sq_r)  # augmented version of gaussian
        lqr.mvn_xi_c = prod_c.concatenate_gaussian(sq_c)  # augmented version of gaussian
        lqr.mvn_u = -4  # N(0, R_s) R_s: R^{DTxDT}
        lqr.x0_l = self.repr_l.demos_xdx[show_demo_idx][0]  # zeta_0 R^DC => 4
        lqr.x0_r = self.repr_r.demos_xdx[show_demo_idx][0]
        lqr.x0_c = self.repr_c.demos_xdx[show_demo_idx][0]

        xi_l, xi_r = lqr.seq_xi

        if self.plot:
            plt.figure()
            plt.title('Trajectory reproduction')
            plt.plot(xi_l[:, 0], xi_l[:, 1], color='r', lw=2, label='generated left line')
            plt.plot(xi_r[:, 0], xi_r[:, 1], color='b', lw=2, label='generated right line')
            # pbd.plot_gmm(mod1.mu, mod1.sigma, swap=True, ax=ax[0], dim=[0, 1], color='steelblue', alpha=0.3)
            # pbd.plot_gmm(mod2.mu, mod2.sigma, swap=True, ax=ax[0], dim=[0, 1], color='orangered', alpha=0.3)
            # pbd.plot_gmm(prod.mu, prod.sigma, swap=True, dim=[0, 1], color='gold')
            # plt.plot(self.demos_left_x[demo_idx][:, 0], self.demos_left_x[demo_idx][:, 1], 'k--', lw=2, label='demo left line')
            # plt.plot(self.demos_right_x[demo_idx][:, 0], self.demos_right_x[demo_idx][:, 1], 'g--', lw=2, label='demo right line')
            # draw_connect(xi_l, xi_r, '--')

            plt.axis('equal')
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(self.demos_left_x[show_demo_idx, :, 0], self.demos_left_x[show_demo_idx, :, 1], color='darkblue',
                     label='demo left')
            plt.plot(self.demos_right_x[show_demo_idx, :, 0], self.demos_right_x[show_demo_idx, :, 1], color='darkred',
                     label='demo right')
            # draw_connect(self.demos_left_x[demo_idx], self.demos_right_x[demo_idx], '--')
            plt.legend()
            plt.axis('equal')
            plt.show()

        return xi_l, xi_r

    def fit(self) -> Tuple[HMM, HMM, HMM]:
        model_l, model_r = super().fit()
        model_c = self.repr_c.hmm_learning()
        return model_l, model_r, model_c

    def reproduce(self, model_l, model_r, model_c, show_demo_idx: int) -> Tuple[
        ndarray, ndarray, GMM, GMM]:
        prod_l = self.repr_l.poe(model_l, show_demo_idx=show_demo_idx)
        prod_r = self.repr_r.poe(model_r, show_demo_idx=show_demo_idx)
        prod_c = self.repr_c.poe(model_c, show_demo_idx=show_demo_idx)

        if self.plot:
            traj_l = self.repr_l._reproduce(model_l, prod_l, show_demo_idx, self.repr_l.demos_xdx[show_demo_idx][0])
            traj_r = self.repr_r._reproduce(model_r, prod_r, show_demo_idx, self.repr_r.demos_xdx[show_demo_idx][0])
            # # gen_c = generate(model_c, prod_c, self.demos_com_x, self.demos_com_xdx, self.demos_com_xdx_augm)
            # plt.figure()
            plt.plot(traj_l[:, 0], traj_l[:, 1], color='lightblue', label='generated left')
            plt.plot(traj_r[:, 0], traj_r[:, 1], color='lightsalmon', label='generated right')
            # # plt.plot(gen_c[:, 0], gen_c[:, 1], color='lightsalmon', label='generated right')
            # draw_connect(gen_l, gen_r, '--')
            plt.legend()
            plt.axis('equal')
            plt.show()

        # Coordinated trajectory
        ctraj_l, ctraj_r = self._bi_reproduce(model_l, prod_l, model_r, prod_r, model_c, prod_c, show_demo_idx)

        return ctraj_l, ctraj_r, prod_l, prod_r


class TPGMMBiCoord(TPGMMBi):
    """
    TPGMM for bimanual coordination
    """

    def __init__(self, demos_left_x, demos_right_x, nb_states: int = 4, reg: float = 1e-3, horizon: int = 150,
                 plot: bool = False):
        self.P = 3  # Start, end and the other arm
        self.nb_dim = demos_left_x.shape[2]
        self.nb_deriv = 2

        self.demos_left_x = demos_left_x
        self.demos_right_x = demos_right_x
        self.plot = plot

        self.task_params = self.get_rel_task_params()
        self.repr_l = TPGMM(demos_left_x, nb_states, reg, horizon, self.P, plot, self.task_params['left'])
        self.repr_r = TPGMM(demos_right_x, nb_states, reg, horizon, self.P, plot, self.task_params['right'])

    def get_rel_task_params(self) -> dict:
        """
        Get the relative A and b matrices
        :return: A, b
        """
        demos_b_l, demos_b_r = [], []
        demos_A_l, demos_A_r = [], []
        for i in range(len(self.demos_left_x)):
            demos_A_l.append(
                np.tile([np.eye(self.nb_dim)] * self.P, (len(self.demos_left_x[i]), 1, 1, 1)))
            demos_A_r.append(
                np.tile([np.eye(self.nb_dim)] * self.P, (len(self.demos_right_x[i]), 1, 1, 1)))
            demos_b_l.append(
                np.hstack((np.tile(np.vstack([self.demos_left_x[i][0], self.demos_left_x[i][-1]]),
                                   (len(self.demos_left_x[i]), 1, 1)), np.expand_dims(self.demos_right_x[i], axis=1))))
            demos_b_r.append(
                np.hstack((np.tile(np.vstack([self.demos_right_x[i][0], self.demos_right_x[i][-1]]),
                                   (len(self.demos_right_x[i]), 1, 1)), np.expand_dims(self.demos_left_x[i], axis=1))))

        demos_A_xdx_l = [np.kron(np.eye(self.nb_deriv), demos_A_l[i]) for i in range(len(demos_A_l))]
        demos_A_xdx_r = [np.kron(np.eye(self.nb_deriv), demos_A_r[i]) for i in range(len(demos_A_r))]
        demos_b_xdx_l = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b_l]
        demos_b_xdx_r = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b_r]

        task_params = {'left': {'demos_A': demos_A_l, 'demos_b': demos_b_l,
                                'demos_A_xdx': demos_A_xdx_l, 'demos_b_xdx': demos_b_xdx_l},
                       'right': {'demos_A': demos_A_r, 'demos_b': demos_b_r,
                                 'demos_A_xdx': demos_A_xdx_r, 'demos_b_xdx': demos_b_xdx_r}}
        return task_params

    def fit(self) -> Tuple[HMM, HMM]:
        model_l, model_r = super().fit()
        return model_l, model_r
