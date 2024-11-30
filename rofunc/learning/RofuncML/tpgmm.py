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

from typing import Union, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import ndarray

import rofunc as rf
from rofunc.learning.RofuncML.gmm import GMM
from rofunc.learning.RofuncML.hmm import HMM
from rofunc.utils.logger.beauty_logger import beauty_print


class TPGMM:
    def __init__(self, demos_x: Union[List, np.ndarray], task_params: dict, nb_states: int = 4, nb_frames: int = None,
                 reg: float = 1e-3, plot: bool = False, save: bool = False, save_params: dict = None):
        """
        Task-parameterized Gaussian Mixture Model (TP-GMM)
        :param demos_x: demo displacement
        :param task_params: task parameters, require {'frame_origins': [start_xdx, end_xdx]}
        :param nb_states: number of states in the HMM
        :param nb_frames: number of candidate frames in a task-parameterized mixture
        :param reg: regularization coefficient
        :param plot: whether to plot the result
        :param save: whether to save the result
        :param save_params: save parameters, {'save_dir': 'path/to/save', 'save_format': 'eps'}
        """
        self.demos_x = demos_x
        self.nb_states = nb_states
        self.reg = reg
        self.horizon = len(demos_x[0])
        self.plot = plot
        self.save = save
        self.save_params = save_params
        if self.save:
            if 'save_dir' not in self.save_params or 'format' not in self.save_params:
                raise ValueError('Please specify the save dir and format!')
            beauty_print(
                'Save dir: {}, format: {}'.format(self.save_params['save_dir'], self.save_params['format']),
                type='info')

        self.nb_dim = len(self.demos_x[0][0])
        self.nb_deriv = 2  # TODO: for now it is just original state and its first derivative
        self.task_params = task_params
        self.nb_frames = len(task_params['frame_origins']) if nb_frames is None else nb_frames

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
            self.demos_xdx_f, self.demos_xdx_augm = self.get_related_matrix()

    def get_dx(self):
        demos_dx = []
        for i in range(len(self.demos_x)):
            demo_dx = []
            for j in range(len(self.demos_x[i])):
                if 0 < j < len(self.demos_x[i]) - 1:
                    dx = (self.demos_x[i][j + 1] - self.demos_x[i][j - 1]) / 2
                elif j == len(self.demos_x[i]) - 1:  # final state
                    # dx = self.demos_x[i][j] - self.demos_x[i][j - 1]
                    dx = np.zeros(self.nb_dim)
                else:  # initial state j = 0
                    # dx = self.demos_x[i][j + 1] - self.demos_x[i][j]
                    dx = np.zeros(self.nb_dim)
                dx = dx / 0.01
                demo_dx.append(dx)
            demos_dx.append(np.array(demo_dx))
        return demos_dx

    def get_A_b(self):
        """
        Get transformation matrices from custom task parameters
        :return: demos_A, demos_b, demos_A_xdx, demos_b_xdx
        """
        task_params = self.task_params
        if 'demos_A_xdx' in task_params and 'demos_b_xdx' in task_params:
            demos_A, demos_b, demos_A_xdx, demos_b_xdx = task_params['demos_A'], task_params['demos_b'], task_params[
                'demos_A_xdx'], task_params['demos_b_xdx']
        elif 'frame_origins' in task_params:
            frame_origins = task_params['frame_origins']
            demos_b = []
            demos_A = []
            for i in range(len(frame_origins[0])):  # TODO： check this
                demos_b.append(np.tile(np.vstack([frame[i][: self.nb_dim] for frame in frame_origins]),
                                       (len(self.demos_x[i]), 1, 1)))
                demos_A.append(np.tile([np.eye(self.nb_dim)] * self.nb_frames, (len(self.demos_x[i]), 1, 1, 1)))
            demos_A_xdx = [np.kron(np.eye(self.nb_deriv), demos_A[i]) for i in range(len(demos_A))]
            demos_b_xdx = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b]
        else:
            raise ValueError('Invalid task_params')

        self.demo_A = demos_A
        self.demo_b = demos_b
        self.demo_A_xdx = demos_A_xdx
        self.demo_b_xdx = demos_b_xdx
        return self.demo_A, self.demo_b, self.demo_A_xdx, self.demo_b_xdx

    def get_related_matrix(self):
        demos_dx = self.get_dx()
        demos_A, demos_b, demos_A_xdx, demos_b_xdx = self.get_A_b()

        demos_xdx = [np.hstack([_x, _dx]) for _x, _dx in
                     zip(self.demos_x, demos_dx)]  # Position and velocity (nb_points, 14)
        demos_xdx_f = [np.einsum('taji,taj->tai', _A, _x[:, None] - _b) for _x, _A, _b in
                       zip(demos_xdx, demos_A_xdx, demos_b_xdx)]
        demos_xdx_augm = [d.reshape(-1, self.nb_deriv * self.nb_frames * self.nb_dim) for d in
                          demos_xdx_f]  # (nb_points, 14 * nb_frames)
        return demos_dx, demos_A, demos_b, demos_A_xdx, demos_b_xdx, demos_xdx, demos_xdx_f, demos_xdx_augm

    def hmm_learning(self) -> HMM:
        beauty_print('{}'.format('learning HMM model'), type='info')

        model = HMM(nb_states=self.nb_states)
        model.init_hmm_kbins(self.demos_xdx_augm)  # initializing model
        model.em(self.demos_xdx_augm, reg=self.reg)

        fig = rf.RofuncML.hmm_plot(self.nb_dim, self.demos_xdx_f, model, self.task_params)
        if self.save:
            rf.visualab.save_img(fig, self.save_params['save_dir'], format=self.save_params['format'])
        if self.plot:
            plt.show()
        return model

    def poe(self, model: HMM, show_demo_idx: int) -> GMM:
        """
        Product of Expert/Gaussian (PoE), which calculates the mixture distribution from multiple coordinates
        :param model: learned model
        :param show_demo_idx: index of the specific demo to be reproduced
        :return: The product of experts
        """
        mod_list = []

        # get transformation for given demonstration.
        # We use the transformation of the first timestep as they are constant
        if len(self.task_params['frame_origins'][0]) == 1:  # For new task parameters generation
            A, b = self.demo_A_xdx[0][0], self.demo_b_xdx[0][0]  # Attention: here we use self.demo_A_xdx not
            # self.demos_A_xdx, because we just called get_A_b not get_related_matrix
        else:
            A, b = self.demos_A_xdx[show_demo_idx][0], self.demos_b_xdx[show_demo_idx][0]

        for p in range(self.nb_frames):
            # transformed model for coordinate system p
            mod_list.append(model.marginal_model(slice(p * len(b[0]), (p + 1) * len(b[0]))).lintrans(A[p], b[p]))

        # product
        prod = mod_list[0]
        for p in range(1, self.nb_frames):
            prod *= mod_list[p]

        fig = rf.RofuncML.poe_plot(self.nb_dim, mod_list, prod, self.demos_x, show_demo_idx, self.task_params)
        if self.save:
            rf.visualab.save_img(fig, self.save_params['save_dir'], format=self.save_params['format'])
        if self.plot:
            plt.show()
        return prod

    def _reproduce(self, model: HMM, prod: GMM, show_demo_idx: int, start_xdx: np.ndarray, title: str = None,
                   label: str = None) -> np.ndarray:
        """
        Reproduce the specific demo_idx from the learned model

        :param model: learned model
        :param prod: result of PoE
        :param show_demo_idx: index of the specific demo to be reproduced
        :param start_xdx: start state
        :param title: title of the plot
        :param label: label of the plot
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

        fig = rf.RofuncML.gen_plot(self.nb_dim, xi, prod, self.demos_x, show_demo_idx, title=title, label=label)
        if self.save:
            rf.visualab.save_img(fig, self.save_params['save_dir'], format=self.save_params['format'])
        if self.plot:
            plt.show()
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
        traj = self._reproduce(model, prod, show_demo_idx, self.demos_xdx[show_demo_idx][0],
                               title="Trajectory reproduction", label="reproduced line")
        return traj, prod

    def generate(self, model: HMM, ref_demo_idx: int) -> Tuple[np.ndarray, GMM]:
        """
        Generate a new trajectory from the learned model
        """
        beauty_print('generate a trajectory from learned representation with new task parameters', type='info')

        self.get_A_b()

        prod = self.poe(model, ref_demo_idx)
        traj = self._reproduce(model, prod, ref_demo_idx, self.task_params['frame_origins'][0][0],
                               title="Trajectory generation", label="generated line")
        return traj, prod


class TPGMMBi(TPGMM):
    """
    Simple TPGMMBi (no coordination)
    """

    def __init__(self, demos_left_x: Union[List, np.ndarray], demos_right_x: Union[List, np.ndarray], task_params: dict,
                 nb_states: int = 4, reg: float = 1e-3, plot: bool = False, save: bool = False,
                 save_params: dict = None):
        assert len(demos_left_x) == len(
            demos_right_x), 'The number of demonstrations for left and right arm should be the same'
        assert len(demos_left_x[0]) == len(
            demos_right_x[0]), 'The number of timesteps for left and right arm should be the same'
        self.demos_left_x = demos_left_x
        self.demos_right_x = demos_right_x
        self.nb_dim = len(demos_left_x[0][0])
        self.plot = plot
        self.save = save
        self.save_params = save_params

        self.repr_l = TPGMM(demos_left_x, task_params['left'], nb_states=nb_states, reg=reg, plot=plot,
                            save=save, save_params=save_params)
        self.repr_r = TPGMM(demos_right_x, task_params['right'], nb_states=nb_states, reg=reg, plot=plot,
                            save=save, save_params=save_params)

    def fit(self) -> Tuple[HMM, HMM]:
        """
        Learning the single arm/agent trajectory representation from demonstration via TP-GMM.
        """
        beauty_print('Learning the bimanual trajectory representation from demonstration via TP-GMM')

        model_l = self.repr_l.hmm_learning()
        model_r = self.repr_r.hmm_learning()
        return model_l, model_r

    def reproduce(self, models: List, show_demo_idx: int) -> Tuple[ndarray, ndarray, GMM, GMM]:
        """
        Reproduce the specific demo_idx from the learned model
        """
        beauty_print('reproduce {}-th demo from learned representation'.format(show_demo_idx), type='info')

        model_l, model_r = models

        prod_l = self.repr_l.poe(model_l, show_demo_idx)
        prod_r = self.repr_r.poe(model_r, show_demo_idx)
        traj_l = self.repr_l._reproduce(model_l, prod_l, show_demo_idx, self.repr_l.demos_xdx[show_demo_idx][0])
        traj_r = self.repr_r._reproduce(model_r, prod_r, show_demo_idx, self.repr_r.demos_xdx[show_demo_idx][0])

        data_lst = [traj_l[:, :self.repr_l.nb_dim], traj_r[:, :self.repr_r.nb_dim]]
        fig = rf.visualab.traj_plot(data_lst, title='Reproduced bimanual trajectories')
        if self.save:
            rf.visualab.save_img(fig, self.save_params['save_dir'], format=self.save_params['format'])
        if self.plot:
            plt.show()
        return traj_l, traj_r, prod_l, prod_r

    def generate(self, models: List, ref_demo_idx: int) -> Tuple[ndarray, ndarray, GMM, GMM]:
        """
        Generate a new trajectory from the learned model
        """
        beauty_print('generate trajectories from learned representation with new task parameters', type='info')

        model_l, model_r = models

        self.repr_l.task_params = self.task_params['left']
        self.repr_r.task_params = self.task_params['right']

        self.repr_l.get_A_b()
        self.repr_r.get_A_b()

        prod_l = self.repr_l.poe(model_l, ref_demo_idx)
        prod_r = self.repr_r.poe(model_r, ref_demo_idx)
        traj_l = self.repr_l._reproduce(model_l, prod_l, ref_demo_idx, self.repr_l.task_params['frame_origins'][0][0])
        traj_r = self.repr_r._reproduce(model_r, prod_r, ref_demo_idx, self.repr_r.task_params['frame_origins'][0][0])

        data_lst = [traj_l[:, :self.repr_l.nb_dim], traj_r[:, :self.repr_r.nb_dim]]
        fig = rf.visualab.traj_plot(data_lst, title='Generated bimanual trajectories')
        if self.save:
            rf.visualab.save_img(fig, self.save_params['save_dir'], format=self.save_params['format'])
        if self.plot:
            plt.show()
        return traj_l, traj_r, prod_l, prod_r


class TPGMM_RPCtrl(TPGMMBi):
    """
    Simple TPGMMBi (no coordination in the representation) with bimanual coordination in the LQR controller
    """

    def __init__(self, demos_left_x, demos_right_x, task_params, nb_states: int = 4, reg: float = 1e-3,
                 plot: bool = False, save: bool = False, save_params: dict = None):
        super().__init__(demos_left_x, demos_right_x, task_params, nb_states=nb_states, reg=reg, plot=plot,
                         save=save, save_params=save_params)

        self.demos_com_x = self.get_rel_demos()
        start_xdx = [self.demos_com_x[i][0] for i in range(len(self.demos_com_x))]  # TODO: change to xdx
        end_xdx = [self.demos_com_x[i][-1] for i in range(len(self.demos_com_x))]
        task_params_c = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
        self.repr_c = TPGMM(self.demos_com_x, task_params_c, nb_states=nb_states, reg=reg, plot=plot, save=save,
                            save_params=save_params)

    def get_rel_demos(self):
        # calculate the relative movement of each demo
        rel_demos = []
        for i in range(len(self.demos_left_x)):
            rel_demo = np.zeros_like(self.demos_left_x[i])
            for j in range(len(self.demos_left_x[i])):
                # rel_demos[i, j] = np.linalg.norm(left_x[i, j] - right_x[i, j])
                rel_demo[j] = self.demos_left_x[i][j] - self.demos_right_x[i][j]
            rel_demos.append(rel_demo)
        return rel_demos

    def _bi_reproduce(self, model_l, prod_l, model_r, prod_r, model_c, prod_c, show_demo_idx, start_xdx_lst):
        """
        Reproduce the specific demo_idx from the learned model

        :param model_l: learned model for left arm
        :param prod_l: result of PoE for left arm
        :param model_r: learned model for right arm
        :param prod_r: result of PoE for ri+588/5/8ght arm
        :param model_c: learned model for relative movement
        :param prod_c: result of PoE for relative movement
        :param show_demo_idx: index of the specific demo to be reproduced
        :param start_xdx_lst: the start states for both arms, List
        :return:
        """
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
        lqr.x0_l = start_xdx_lst[0]  # zeta_0 R^DC => 4
        lqr.x0_r = start_xdx_lst[1]
        lqr.x0_c = start_xdx_lst[2]

        xi_l, xi_r = lqr.seq_xi
        return xi_l, xi_r

    def fit(self) -> Tuple[HMM, HMM, HMM]:
        model_l, model_r = super().fit()
        model_c = self.repr_c.hmm_learning()
        return model_l, model_r, model_c

    def reproduce(self, models, show_demo_idx: int) -> Tuple[ndarray, ndarray, GMM, GMM]:
        """
        Reproduce the specific demo_idx from the learned model

        :param models: List of learned models for left, right and relative movement
        :param show_demo_idx: index of the specific demo to be reproduced
        :return:
        """
        beauty_print('reproduce {}-th demo from learned representation'.format(show_demo_idx), type='info')

        model_l, model_r, model_c = models

        prod_l = self.repr_l.poe(model_l, show_demo_idx=show_demo_idx)
        prod_r = self.repr_r.poe(model_r, show_demo_idx=show_demo_idx)
        prod_c = self.repr_c.poe(model_c, show_demo_idx=show_demo_idx)
        # Coordinated trajectory
        ctraj_l, ctraj_r = self._bi_reproduce(model_l, prod_l, model_r, prod_r, model_c, prod_c, show_demo_idx,
                                              start_xdx_lst=[self.repr_l.demos_xdx[show_demo_idx][0],
                                                             self.repr_r.demos_xdx[show_demo_idx][0],
                                                             self.repr_c.demos_xdx[show_demo_idx][0]])

        data_lst = [ctraj_l[:, :self.repr_l.nb_dim], ctraj_r[:, :self.repr_r.nb_dim]]
        fig = rf.visualab.traj_plot(data_lst, title='Reproduced bimanual trajectories')
        if self.save:
            rf.visualab.save_img(fig, self.save_params['save_dir'], format=self.save_params['format'])
        if self.plot:
            plt.show()
        return ctraj_l, ctraj_r, prod_l, prod_r

    def generate(self, models: List, ref_demo_idx: int) -> Tuple[ndarray, ndarray, GMM, GMM]:
        """
        Generate a new trajectory from the learned model

        :param models: List of learned models for left, right and relative movement
        :param ref_demo_idx: index of the specific demo to be referenced
        :return:
        """
        beauty_print('generate trajectories from learned representation with new task parameters', type='info')

        model_l, model_r, model_c = models

        self.repr_l.task_params = self.task_params['left']
        self.repr_r.task_params = self.task_params['right']
        self.repr_c.task_params = self.task_params['relative']

        self.repr_l.get_A_b()
        self.repr_r.get_A_b()
        self.repr_c.get_A_b()

        prod_l = self.repr_l.poe(model_l, ref_demo_idx)
        prod_r = self.repr_r.poe(model_r, ref_demo_idx)
        prod_c = self.repr_c.poe(model_c, ref_demo_idx)

        ctraj_l, ctraj_r = self._bi_reproduce(model_l, prod_l, model_r, prod_r, model_c, prod_c, ref_demo_idx,
                                              # start_xdx_lst=[self.repr_l.task_params['frame_origins'][0][0],
                                              #                self.repr_r.task_params['frame_origins'][0][0],
                                              #                self.repr_c.task_params['frame_origins'][0][0]])
                                              start_xdx_lst=[self.repr_l.demos_xdx[ref_demo_idx][0],
                                                             self.repr_r.demos_xdx[ref_demo_idx][0],
                                                             self.repr_c.demos_xdx[ref_demo_idx][0]])

        data_lst = [ctraj_l[:, :self.repr_l.nb_dim], ctraj_r[:, :self.repr_r.nb_dim]]
        fig = rf.visualab.traj_plot(data_lst, title='Generated bimanual trajectories')
        if self.save:
            rf.visualab.save_img(fig, self.save_params['save_dir'], format=self.save_params['format'])
        if self.plot:
            plt.show()
        return ctraj_l, ctraj_r, prod_l, prod_r


class TPGMM_RPRepr(TPGMMBi):
    """
    TPGMM for bimanual coordination in representation
    """

    def __init__(self, demos_left_x, demos_right_x, task_params, nb_states: int = 4, reg: float = 1e-3,
                 plot: bool = False, save: bool = False, save_params: dict = None, **kwargs):
        self.demos_left_x = demos_left_x
        self.demos_right_x = demos_right_x
        self.nb_dim = len(demos_left_x[0][0])
        self.plot = plot
        self.save = save
        self.save_params = save_params
        self.nb_frames = len(task_params['left']['frame_origins']) + 1  # Additional observe frame from the other arm
        self.nb_deriv = 2
        self.nb_states = nb_states
        self.task_params = task_params

        self.get_rel_task_params()
        self.repr_l = TPGMM(demos_left_x, self.task_params['left'], nb_states=nb_states, nb_frames=self.nb_frames,
                            reg=reg, plot=plot, save=save, save_params=save_params)
        self.repr_r = TPGMM(demos_right_x, self.task_params['right'], nb_states=nb_states, nb_frames=self.nb_frames,
                            reg=reg, plot=plot, save=save, save_params=save_params)

    def get_rel_task_params(self):
        demos_b_l, demos_b_r = [], []
        demos_A_l, demos_A_r = [], []

        frame_origins_l = self.task_params['left']['frame_origins']
        frame_origins_r = self.task_params['right']['frame_origins']
        for i in range(len(frame_origins_l[0])):
            demos_A_l.append(np.tile([np.eye(self.nb_dim)] * self.nb_frames, (len(self.demos_left_x[i]), 1, 1, 1)))
            demos_A_r.append(np.tile([np.eye(self.nb_dim)] * self.nb_frames, (len(self.demos_right_x[i]), 1, 1, 1)))
            demos_b_l.append(np.hstack((np.tile(np.vstack([frame[i][: self.nb_dim] for frame in frame_origins_l]),
                                                (len(self.demos_left_x[i]), 1, 1)),
                                        np.expand_dims(self.demos_right_x[i], axis=1))))
            demos_b_r.append(np.hstack((np.tile(np.vstack([frame[i][: self.nb_dim] for frame in frame_origins_r]),
                                                (len(self.demos_right_x[i]), 1, 1)),
                                        np.expand_dims(self.demos_left_x[i], axis=1))))

        demos_A_xdx_l = [np.kron(np.eye(self.nb_deriv), demos_A_l[i]) for i in range(len(demos_A_l))]
        demos_A_xdx_r = [np.kron(np.eye(self.nb_deriv), demos_A_r[i]) for i in range(len(demos_A_r))]
        demos_b_xdx_l = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b_l]
        demos_b_xdx_r = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b_r]

        self.task_params = {
            'left': {'frame_names': self.task_params['left']['frame_names'] + ['relative'],
                     'demos_A': demos_A_l, 'demos_b': demos_b_l,
                     'demos_A_xdx': demos_A_xdx_l, 'demos_b_xdx': demos_b_xdx_l},
            'right': {'frame_names': self.task_params['right']['frame_names'] + ['relative'],
                      'demos_A': demos_A_r, 'demos_b': demos_b_r,
                      'demos_A_xdx': demos_A_xdx_r, 'demos_b_xdx': demos_b_xdx_r}}

    def _get_dyna_A_b(self, model, repr, show_demo_idx, task_params=None):
        ref_frame = 0  # TODO: 0 for start frame, 1 for end frame, 2 for the relative frame

        if task_params is not None:
            repr.P = 2
            demos_dx, demos_A, demos_b, demos_A_xdx, demos_b_xdx, demos_xdx, demos_xdx_f, demos_xdx_augm = repr.get_related_matrix(
                task_params)
            repr.P = 3

            # demos_xdx_f = [
            #     np.concatenate([demos_xdx_f[0], np.expand_dims(repr.demos_xdx_f[show_demo_idx][:, 2], axis=1)], axis=1)]
            demos_xdx_f = repr.demos_xdx_f
            demos_A_xdx = [
                np.concatenate([demos_A_xdx[0], np.expand_dims(repr.demos_A_xdx[show_demo_idx][:, 2], axis=1)], axis=1)]
            demos_b_xdx = [
                np.concatenate([demos_b_xdx[0], np.expand_dims(repr.demos_b_xdx[show_demo_idx][:, 2], axis=1)], axis=1)]
            show_demo_idx = 0
        else:
            demos_xdx_f = repr.demos_xdx_f
            demos_A_xdx = repr.demos_A_xdx
            demos_b_xdx = repr.demos_b_xdx

        mus = model._mu[:,
              self.nb_dim * self.nb_deriv * ref_frame:self.nb_dim * self.nb_deriv * ref_frame + self.nb_dim]
        demo_traj_in_ref_frame = demos_xdx_f[show_demo_idx][:, ref_frame, :self.nb_dim]
        index_list = []
        for mu in mus:
            dist = list(scipy.spatial.distance.cdist([mu], demo_traj_in_ref_frame)[0])
            index = dist.index(min(dist))
            index_list.append(index)

        A, b = demos_A_xdx[show_demo_idx][index_list], demos_b_xdx[show_demo_idx][index_list]
        return A, b, index_list

    def _uni_poe(self, model, repr, show_demo_idx, task_params=None) -> GMM:
        if task_params is None:
            A, b, _ = self._get_dyna_A_b(model, repr, show_demo_idx)
        else:
            A, b = task_params['A'], task_params['b']
        mod_list = []
        for p in range(self.nb_frames):
            mod_list.append(
                model.marginal_model(
                    slice(p * self.nb_deriv * self.nb_dim, (p + 1) * self.nb_deriv * self.nb_dim)).lintrans_dyna(
                    A[:, p], b[:, p]))
        # product
        # prod = mod_list[0]
        # for p in range(1, self.nb_frames):
        #     prod *= mod_list[p]

        # Weighted product
        prod = mod_list[0] * mod_list[1] * mod_list[2] * mod_list[2] * mod_list[2] * mod_list[2] * mod_list[2] * \
               mod_list[2] * mod_list[2]

        fig = rf.RofuncML.poe_plot(self.nb_dim, mod_list, prod, repr.demos_x, show_demo_idx, repr.task_params)
        if self.save:
            rf.visualab.save_img(fig, self.save_params['save_dir'], format=self.save_params['format'])
        if self.plot:
            plt.show()
        return prod

    def _bi_poe(self, models: List, show_demo_idx: int) -> Tuple[GMM, GMM]:
        """
        Product of Expert/Gaussian (PoE), which calculates the mixture distribution from multiple coordinates
        :param models: list of pre_trained_models
        :param show_demo_idx: index of the specific demo to be reproduced
        :return: The product of experts
        """
        model_l, model_r = models

        prod_l = self._uni_poe(model_l, self.repr_l, show_demo_idx)
        prod_r = self._uni_poe(model_r, self.repr_r, show_demo_idx)
        return prod_l, prod_r

    def fit(self) -> Tuple[HMM, HMM]:
        model_l, model_r = super().fit()
        return model_l, model_r

    def reproduce(self, models: List, show_demo_idx: int) -> Tuple[ndarray, ndarray, GMM, GMM]:
        beauty_print('reproduce {}-th demo from learned representation'.format(show_demo_idx), type='info')

        model_l, model_r = models

        prod_l, prod_r = self._bi_poe([model_l, model_r], show_demo_idx)
        ctraj_l = self.repr_l._reproduce(model_l, prod_l, show_demo_idx, self.repr_l.demos_xdx[show_demo_idx][0])
        ctraj_r = self.repr_r._reproduce(model_r, prod_r, show_demo_idx, self.repr_r.demos_xdx[show_demo_idx][0])

        data_lst = [ctraj_l[:, :self.nb_dim], ctraj_r[:, :self.nb_dim]]
        fig = rf.visualab.traj_plot(data_lst, title='Reproduced bimanual trajectories')
        if self.save:
            rf.visualab.save_img(fig, self.save_params['save_dir'], format=self.save_params['format'])
        if self.plot:
            plt.show()
        return ctraj_l, ctraj_r, prod_l, prod_r

    def iterative_generate(self, model_l: HMM, model_r: HMM, ref_demo_idx: int, nb_iter=1) -> \
            Tuple[ndarray, ndarray, GMM, GMM]:
        beauty_print('generate trajectories from learned representation with new task parameters iteratively',
                     type='info')

        vanilla_repr = TPGMMBi(self.demos_left_x, self.demos_right_x, task_params=self.task_params,
                               nb_states=self.nb_states, plot=self.plot, save=self.save, save_params=self.save_params)
        vanilla_repr.task_params = self.task_params
        vanilla_model_l, vanilla_model_r = vanilla_repr.fit()

        vanilla_traj_l, vanilla_traj_r, _, _ = vanilla_repr.generate([vanilla_model_l, vanilla_model_r],
                                                                     ref_demo_idx=ref_demo_idx)

        # _, _, _, _, task_params_A_b_l = vanilla_repr.repr_l.get_A_b_from_task_params(task_params['left'])
        # _, _, _, _, task_params_A_b_r = vanilla_repr.repr_r.get_A_b_from_task_params(task_params['right'])
        #
        # # Generate without coordination originally
        # vanilla_prod_l = vanilla_repr.repr_l.poe(vanilla_model_l, ref_demo_idx, task_params_A_b_l)
        # vanilla_prod_r = vanilla_repr.repr_r.poe(vanilla_model_r, ref_demo_idx, task_params_A_b_r)
        # vanilla_traj_l = vanilla_repr.repr_l._reproduce(vanilla_model_l, vanilla_prod_l, ref_demo_idx,
        #                                                 task_params['left']['start_xdx'])
        # vanilla_traj_r = vanilla_repr.repr_r._reproduce(vanilla_model_r, vanilla_prod_r, ref_demo_idx,
        #                                                 task_params['right']['start_xdx'])

        data_lst = [vanilla_traj_l[:, :self.nb_dim], vanilla_traj_r[:, :self.nb_dim]]
        fig = rf.visualab.traj_plot(data_lst, title='Generate Trajectories without Coordination')
        if self.save:
            rf.visualab.save_img(fig, self.save_params['save_dir'], format=self.save_params['format'])
        if self.plot:
            plt.show()

        # Add coordination to generate the trajectories iteratively
        traj_l, traj_r = vanilla_traj_l, vanilla_traj_r
        for i in range(nb_iter):

            self.task_params['left']['traj'] = traj_l[:, :self.nb_dim]
            _, ctraj_r, _, prod_r = self.conditional_generate(model_l, model_r, ref_demo_idx, leader='left')
            self.task_params['right']['traj'] = traj_r[:, :self.nb_dim]
            _, ctraj_l, _, prod_l = self.conditional_generate(model_l, model_r, ref_demo_idx, leader='right')

            traj_l, traj_r = ctraj_l, ctraj_r

            data_lst = [traj_l[:, :self.nb_dim], traj_r[:, :self.nb_dim]]
            fig = rf.visualab.traj_plot(data_lst, title='Generated bimanual trajectories in synergistic manner,'
                                                        ' iteration: {}-th'.format(i + 1))
            if self.save:
                rf.visualab.save_img(fig, self.save_params['save_dir'], format=self.save_params['format'])
            if self.plot:
                plt.show()

        return ctraj_l, ctraj_r, prod_l, prod_r

    def conditional_generate(self, model_l: HMM, model_r: HMM, ref_demo_idx: int, leader: str) -> \
            Tuple[ndarray, ndarray, None, GMM]:
        follower = 'left' if leader == 'right' else 'right'
        leader_traj = self.task_params[leader]['traj']
        models = {'left': model_l, 'right': model_r}
        reprs = {'left': self.repr_l, 'right': self.repr_r}

        beauty_print('generate the {} trajectory from learned representation conditioned on the {} trajectory'.format(
            follower, leader), type='info')

        A, b, index_list = self._get_dyna_A_b(models[follower], reprs[follower], ref_demo_idx,
                                              task_params=self.task_params[follower])
        b[:, 2, :self.nb_dim] = leader_traj[index_list, :self.nb_dim]

        follower_prod = self._uni_poe(models[follower], reprs[follower], ref_demo_idx, task_params={'A': A, 'b': b})

        follower_traj = reprs[follower]._reproduce(models[follower], follower_prod, ref_demo_idx,
                                                   self.task_params[follower]['start_xdx'])

        data_lst = [leader_traj[:, :self.nb_dim], follower_traj[:, :self.nb_dim]]
        fig = rf.visualab.traj_plot(data_lst, title='Generated bimanual trajectories in leader-follower manner')
        if self.save:
            rf.visualab.save_img(fig, self.save_params['save_dir'], format=self.save_params['format'])
        if self.plot:
            plt.show()
        return leader_traj, follower_traj, None, follower_prod

    def generate(self, models: List, ref_demo_idx: int, leader: str = None):
        model_l, model_r = models
        if leader is None:
            return self.iterative_generate(model_l, model_r, ref_demo_idx)
        else:
            return self.conditional_generate(model_l, model_r, ref_demo_idx, leader)


class TPGMM_RPAll(TPGMM_RPRepr, TPGMM_RPCtrl):
    """
    TPGMM for bimanual coordination in both representation and LQR controller
    """

    def __init__(self, demos_left_x, demos_right_x, nb_states: int = 4, reg: float = 1e-3, horizon: int = 150,
                 plot: bool = False, save: bool = False, save_params: dict = None, **kwargs):
        TPGMM_RPRepr.__init__(self, demos_left_x, demos_right_x, nb_states, reg, horizon, plot=plot, save=save,
                              save_params=save_params, **kwargs)

        self.demos_com_x = self.get_rel_demos()
        self.repr_c = TPGMM(self.demos_com_x, nb_states=nb_states, reg=reg, horizon=horizon, plot=plot, save=save,
                            save_params=save_params, **kwargs)

    def fit(self):
        return TPGMM_RPCtrl.fit(self)

    def reproduce(self, model_l: HMM, model_r: HMM, model_c: HMM, show_demo_idx: int) -> Tuple[
        ndarray, ndarray, GMM, GMM]:
        beauty_print('reproduce {}-th demo from learned representation'.format(show_demo_idx), type='info')

        prod_l, prod_r = self._bi_poe(model_l, model_r, show_demo_idx)
        prod_c = self.repr_c.poe(model_c, show_demo_idx=show_demo_idx)

        # Coordinated trajectory
        ctraj_l, ctraj_r = self._bi_reproduce(model_l, prod_l, model_r, prod_r, model_c, prod_c, show_demo_idx)
        return ctraj_l, ctraj_r, prod_l, prod_r

    def generate(self, model_l: HMM, model_r: HMM, model_c: HMM, ref_demo_idx: int, task_params: dict,
                 leader: str = None):
        # TODO
        if leader is None:
            _, _, prod_l, prod_r = self.iterative_generate(model_l, model_r, ref_demo_idx, task_params)
            prod_c = self.repr_c.poe(model_c, show_demo_idx=ref_demo_idx)
            traj_l, traj_r = self._bi_reproduce(model_l, prod_l, model_r, prod_r, model_c, prod_c, ref_demo_idx)
            return traj_l, traj_r, None, None
        else:
            return self.conditional_generate(model_l, model_r, ref_demo_idx, task_params, leader)
