import copy
from typing import Union, List, Tuple

import numpy as np
import pbdlib as pbd

import rofunc as rf
from rofunc.utils.logger.beauty_logger import beauty_print


class TPHSMM:
    def __init__(self, demos: Union[List, np.ndarray], nb_states: int = 4, reg: float = 1e-3, horizon: int = 150,
                 plot: bool = False, task_params: Union[List, Union[List, Union[Tuple, np.ndarray]]] = None,
                 dt: float = 0.01):
        """
        Task-parameterized Hidden Semi-Markov Model (TP-GMM)
        :param demos: demo displacement
        :param nb_states: number of states in the HMM
        :param reg: regularization coefficient
        :param horizon: horizon of the reproduced trajectory
        :param plot: whether to plot the result
        """

        self.demos_x = demos
        self.nb_states = nb_states
        self.reg = reg
        self.horizon = horizon
        self.plot = plot
        self.task_params = task_params
        self.dt = dt
        self.hsmm = None
        self.demos_xdx = [np.concatenate([x, dx], axis=-1) for x, dx in zip(self.demos_x, self.get_dx(self.demos_x))]
        self.nb_dim = self.demos_xdx[0].shape[1]

        # TODO Handle case of moving task parameters ?
        self.n_tp = task_params[0][0].shape[0]

        self.demos_tp_f, self.demos_tp = self._task_parametrize(self.demos_xdx, self.task_params)

    def get_dx(self, demos_x):
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
                dx = dx / self.dt
                demo_dx.append(dx)
            demos_dx.append(np.array(demo_dx))
        return demos_dx

    def _task_parametrize(self, demos_xdx, task_params):
        beauty_print(f'[{self.__class__.__name__}] Task parametrization')
        print(task_params[0][0].shape)
        if len(task_params[0][0].shape) == 4:
            demos_tp_f = [np.einsum('atij,atj->tai', _A, _x - _b)
                          for _x, (_A, _b) in zip(demos_xdx, task_params)]
        else:
            print([(_x.shape, _A.shape, _b.shape) for _x, (_A, _b) in zip(demos_xdx, task_params)])
            demos_tp_f = [np.einsum('taij,taj->tai', _A[None, :], _x[:, None] - _b)
                          for _x, (_A, _b) in zip(demos_xdx, task_params)]
        demos_tp = [d.reshape(-1, self.nb_dim * self.n_tp) for d in demos_tp_f]
        return demos_tp_f, demos_tp

    def hsmm_learning(self):
        """
        Learn the task-parameterized HMM
        """
        beauty_print(f'[{self.__class__.__name__}] Learning...', type='info')
        self.hsmm = pbd.HSMM(nb_states=self.nb_states, nb_dim=self.nb_dim)
        self.hsmm.init_hmm_kbins(self.demos_tp)
        self.hsmm.em(self.demos_tp, reg=self.reg)

        if self.plot:
            rf.RofuncML.hmm_plot(self.nb_dim, self.demos_tp_f, self.hsmm)

    def poe(self, show_demo_idx: int, task_params: tuple = None) -> pbd.GMM:
        """
        Product of Expert/Gaussian (PoE), which calculates the mixture distribution from multiple coordinates
        :param model: learned model
        :param show_demo_idx: index of the specific demo to be reproduced
        :param task_params: [dict], task parameters for including transformation matrix A and bias b
        :return: The product of experts
        """
        # get transformation for given demonstration.
        if task_params is not None:
            _A, _b = task_params[0], task_params[1]
        else:
            # We use the transformation of the first timestep as they are constant
            # Are they always constant ? (No)
            _A, _b = self.task_params[show_demo_idx]
        if len(_A.shape) == 4:
            _A, _b = _A[:, 0, :, :], _b[:, 0, :]

        marginal_models = []
        # Get porduct of marginals
        for o in range(_A.shape[0]):
            mod = self.hsmm.marginal_model(slice(o * self.nb_dim, (o + 1) * self.nb_dim)).lintrans(_A[o].T, _b[o])
            # reproj_demos.append([(_A[o].T @ d[:, o * self.nb_dim:(o + 1) * self.nb_dim].T).T + _b[o] for d in demo_frames_aug])
            marginal_models.append(mod)
            if o == 0:
                prod = copy.deepcopy(mod)
            else:
                prod = prod * mod

        if self.plot:
            rf.RofuncML.poe_plot(self.nb_dim, [marginal_models[0], marginal_models[1]], prod, self.demos_x, show_demo_idx)
        return prod

    def _reproduce(self, prod: pbd.GMM, show_demo_idx: int, start_xdx: np.ndarray,
                   dt: float = None) -> np.ndarray:
        """
        Reproduce the specific demo_idx from the learned model
        :param model: learned model
        :param prod: result of PoE
        :param show_demo_idx: index of the specific demo to be reproduced
        :param start_xdx: start state
        :return:
        """
        if dt is None:
            dt = self.dt
        # get the most probable sequence of state for this demonstration
        sq = self.hsmm.viterbi(self.demos_tp[show_demo_idx])
        sq = np.repeat(np.array(sq), int(self.dt / dt))

        # solving LQR with Product of Gaussian, see notebook on LQR
        lqr = rf.lqr.PoGLQR(nb_dim=self.nb_dim // 2, dt=dt,
                            horizon=int(self.demos_xdx[show_demo_idx].shape[0] * self.dt / dt))
        lqr.mvn_xi = prod.concatenate_gaussian(sq)  # augmented version of gaussian
        lqr.mvn_u = -4
        lqr.x0 = start_xdx

        xi = lqr.seq_xi
        if self.plot:
            rf.RofuncML.gen_plot(self.nb_dim, xi, prod, self.demos_x, show_demo_idx)
        return xi

    def _generate(self, prod: pbd.GMM, ref_demo_idx: int,
                  start_x: np.ndarray, task_params: tuple,
                  horizon: int, dt: float = None) -> np.ndarray:
        """
        Reproduce the specific demo_idx from the learned model
        :param model: learned model
        :param prod: result of PoE
        :param show_demo_idx: index of the specific demo to be reproduced
        :param start_xdx: start state
        :return:
        """
        if dt is None:
            dt = self.dt
        # # get the most probable sequence of state for this demonstration
        # sq = self.hsmm.viterbi(self.demos_tp[show_demo_idx])
        start_xdx = np.concatenate([start_x, self.get_dx([start_x])[0]], axis=-1)
        _, start_xdx_tp = self._task_parametrize(start_xdx, task_params=[task_params])
        p0 = self.hsmm.forward_variable(demo=start_xdx_tp[0])[:, 0]
        if np.isnan(p0).any():
            p0 = None
        sq = self.hsmm.forward_variable_ts(horizon, p0=p0)
        sq = np.argmax(sq, axis=0)
        # sq = np.repeat(np.array(sq), int(self.dt / dt))

        # solving LQR with Product of Gaussian, see notebook on LQR
        lqr = rf.lqr.PoGLQR(nb_dim=self.nb_dim // 2, dt=dt, horizon=horizon)
        lqr.mvn_xi = prod.concatenate_gaussian(sq)  # augmented version of gaussian
        lqr.mvn_u = -4
        lqr.x0 = start_xdx[0]

        xi = lqr.seq_xi
        if self.plot:
            rf.RofuncML.gen_plot(self.nb_dim, xi, prod, self.demos_x, ref_demo_idx)
        return xi

    def fit(self) -> pbd.HSMM:
        """
        Learning the single arm/agent trajectory representation from demonstration via TP-HSMM.
        """
        beauty_print('Learning the trajectory representation from demonstration via TP-HSMM')

        model = self.hsmm_learning()
        return model

    def reproduce(self, show_demo_idx: int, dt: float = None) -> Tuple[np.ndarray, pbd.GMM]:
        """
        Reproduce the specific demo_idx from the learned model
        """
        beauty_print('reproduce {}-th demo from learned representation'.format(show_demo_idx), type='info')

        if dt is None:
            dt = self.dt
        prod = self.poe(show_demo_idx)
        traj = self._reproduce(prod, show_demo_idx, self.demos_xdx[show_demo_idx][0], dt=dt)
        return traj, prod

    def generate(self, ref_demo_idx: int, task_params: tuple,
                 start_state: np.array, horizon: int = 100, dt: float = None) -> Tuple[np.ndarray, pbd.GMM]:
        """
        Generate a new trajectory from the learned model
        """
        beauty_print('generate a new demo from learned representation', type='info')

        # task_params_A_b = self.get_task_params_A_b(task_params)
        if dt is None:
            dt = self.dt

        beauty_print(f"[{self.__class__.__name__}] Start state: {start_state}")
        beauty_print(f"[{self.__class__.__name__}] Start state shape: {start_state.shape}")
        prod = self.poe(ref_demo_idx, task_params)
        traj = self._generate(prod, ref_demo_idx, start_state, task_params, horizon, dt=dt)
        return traj, prod
