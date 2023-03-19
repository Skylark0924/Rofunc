import numpy as np
from pbdlib.utils.utils import lifted_transfer_matrix
import pbdlib as pbd
from pbdlib import MVN
from tqdm import tqdm


class LQR(object):
    def __init__(self, A=None, B=None, nb_dim=2, dt=0.01, horizon=50):
        self._horizon = horizon
        self.A = A
        self.B = B
        self.dt = dt

        self.nb_dim = nb_dim

        self._s_xi, self._s_u = None, None
        self._x0 = None

        self._gmm_xi, self._gmm_u = None, None
        self._mvn_sol_xi, self._mvn_sol_u = None, None

        self._seq_xi, self._seq_u = None, None

        self._S, self._v, self._K, self._Kv, self._ds, self._cs, self._Qc = \
            None, None, None, None, None, None, None

        self._Q, self._z = None, None

    @property
    def seq_xi(self):
        return self._seq_xi

    @seq_xi.setter
    def seq_xi(self, value):
        self._seq_xi = value

    @property
    def K(self):
        assert self._K is not None, "Solve Ricatti before"

        return self._K

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, value):
        """
        value :
            (ndim_xi, ndim_xi) or
            ((N, ndim_xi, ndim_xi), (nb_timestep, )) or
            (nb_timestep, ndim_xi, ndim_xi)
        """
        self._Q = value

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        """
        value :
            (ndim_xi, ) or
            ((N, ndim_xi, ), (nb_timestep, )) or
            (nb_timestep, ndim_xi)
        """
        self._z = value

    @property
    def Qc(self):
        assert self._Qc is not None, "Solve Ricatti before"

        return self._Qc

    @property
    def cs(self):
        """
        Return c list where control command u is
            u = -K x + c

        :return:
        """
        if self._cs is None:
            self._cs = self.get_feedforward()

        return self._cs

    @property
    def ds(self):
        """
        Return c list where control command u is
            u = K(d - x)

        :return:
        """
        if self._ds is None:
            self._ds = self.get_target()

        return self._ds

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, value):
        # self.reset_params()

        self._horizon = value

    @property
    def u_dim(self):
        """
        Number of dimension of input
        :return:
        """
        if self.B is not None:
            return self.B.shape[1]
        else:
            return self.nb_dim

    @property
    def xi_dim(self):
        """
        Number of dimension of state
        :return:
        """
        if self.A is not None:
            return self.A.shape[0]
        else:
            return self.nb_dim * 2

    @property
    def gmm_xi(self):
        """
        Distribution of state
        :return:
        """
        return self._gmm_xi

    @gmm_xi.setter
    def gmm_xi(self, value):
        """
        :param value 		[pbd.GMM] or [(pbd.GMM, list)]
        """
        # resetting solution
        self._mvn_sol_xi = None
        self._mvn_sol_u = None
        self._seq_u = None
        self._seq_xi = None

        self._gmm_xi = value

    @property
    def gmm_u(self):
        """
        Distribution of control input
        :return:
        """
        return self._gmm_u

    @gmm_u.setter
    def gmm_u(self, value):
        """
        :param value 		[float] or [pbd.MVN] or [pbd.GMM] or [(pbd.GMM, list)]
        """
        # resetting solution
        self._mvn_sol_xi = None
        self._mvn_sol_u = None
        self._seq_u = None
        # self._seq_xi = None

        if isinstance(value, float):
            self._gmm_u = pbd.MVN(
                mu=np.zeros(self.u_dim), lmbda=10 ** value * np.eye(self.u_dim))
        else:
            self._gmm_u = value

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, value):
        # resetting solution
        self._mvn_sol_xi = None
        self._mvn_sol_u = None

        self._x0 = value

    def get_Q_z(self, t):
        """
        get Q and target z for time t
        :param t:
        :return:
        """
        if self._gmm_xi is None:
            z, Q = None, None

            if self._z is None:
                z = np.zeros(self.A.shape[-1])
            elif isinstance(self._z, tuple):
                z = self._z[0][self._z[1][t]]
            elif isinstance(self._z, np.ndarray):
                if self._z.ndim == 1:
                    z = self._z
                elif self._z.ndim == 2:
                    if self._seq_xi is None:
                        z = self._z[t]
                    else:
                        z = self._z[self._seq_xi[t]]

            if isinstance(self._Q, tuple):
                Q = self._Q[0][self._Q[1][t]]
            elif isinstance(self._Q, np.ndarray):

                if self._Q.ndim == 2:
                    Q = self._Q
                elif self._Q.ndim == 3:
                    if self._seq_xi is None:
                        Q = self._Q[t]
                    else:
                        Q = self._Q[self._seq_xi[t]]

            return Q, z
        else:
            if isinstance(self._gmm_xi, tuple):
                gmm, seq = self._gmm_xi
                return gmm.lmbda[seq[t]], gmm.mu[seq[t]]
            elif isinstance(self._gmm_xi, pbd.GMM):
                return self._gmm_xi.lmbda[t], self._gmm_xi.mu[t]
            elif isinstance(self._gmm_xi, pbd.MVN):
                return self._gmm_xi.lmbda, self._gmm_xi.mu
            else:
                raise ValueError("Not supported gmm_xi")

    def get_R(self, t):
        if isinstance(self._gmm_u, pbd.MVN):
            return self._gmm_u.lmbda
        elif isinstance(self._gmm_u, tuple):
            gmm, seq = self._gmm_u
            return gmm.lmbda[seq[t]]
        elif isinstance(self._gmm_u, pbd.GMM):
            return self._gmm_u.lmbda[t]
        else:
            raise ValueError("Not supported gmm_u")

    def get_A(self, t):
        if self.A.ndim == 2:
            return self.A
        else:
            return self.A[t]

    def get_B(self, t):
        if self.B.ndim == 2:
            return self.B
        else:
            return self.B[t]

    def ricatti(self):
        """
        http://web.mst.edu/~bohner/papers/tlqtots.pdf
        :return:
        """

        Q, z = self.get_Q_z(-1)
        #
        _S = [None for i in range(self._horizon)]
        _v = [None for i in range(self._horizon)]
        _K = [None for i in range(self._horizon - 1)]
        _Kv = [None for i in range(self._horizon - 1)]
        _Qc = [None for i in range(self._horizon - 1)]

        _S[-1] = Q
        _v[-1] = Q.dot(z)

        for t in tqdm(range(self.horizon - 2, -1, -1)):
            Q, z = self.get_Q_z(t)
            R = self.get_R(t)
            A = self.get_A(t)
            B = self.get_B(t)

            _Qc[t] = np.linalg.inv(R + B.T.dot(_S[t + 1]).dot(B))
            _Kv[t] = _Qc[t].dot(B.T)
            _K[t] = _Kv[t].dot(_S[t + 1]).dot(A)

            AmBK = A - B.dot(_K[t])

            _S[t] = A.T.dot(_S[t + 1]).dot(AmBK) + Q
            _v[t] = AmBK.T.dot(_v[t + 1]) + Q.dot(z)

        self._S = _S
        self._v = _v
        self._K = _K
        self._Kv = _Kv
        self._Qc = _Qc

        self._ds = None
        self._cs = None

    def get_target(self):
        ds = []

        for t in range(0, self.horizon - 1):
            ds += [np.linalg.inv(self._S[t].dot(self.A)).dot(self._v[t])]

        return np.array(ds)

    def get_feedforward(self):
        cs = []

        for t in range(0, self.horizon - 1):
            cs += [self._Kv[t].dot(self._v[t + 1])]

        return np.array(cs)

    def get_command(self, xi, i):
        if xi.ndim == 1:
            return -self._K[i].dot(xi) + self._Kv[i].dot(self._v[i])
        else:
            return np.einsum('ij,aj->ai', -self.K[i], xi) + self._Kv[i].dot(self._v[i + 1])

    def policy(self, xi, t):
        """
        Time-dependent and linear in state policy as MVN distribution.
        :param xi: Current state
        :return:
        """
        loc = self.get_command(xi, t)
        Qc = self.Qc[t]
        try:
            np.linalg.cholesky(Qc)
            invertible = True
        except:
            invertible = False
        phi = 1e-8
        while not invertible:
            Qc = Qc + np.eye(self.u_dim) * phi
            try:
                np.linalg.cholesky(Qc)
                invertible = True
            except:
                invertible = False
                phi *= 1E1

        cov = np.tile(Qc[None], (loc.shape[0], 1, 1))

        return MVN(mu=loc, sigma=cov)

    def get_sample(self, xi, i, sample_size=1):
        """

        :param xi:
        :param i:
        :param sample_size:
        :return:
        """
        return self.policy(xi, i).sample(sample_size)

    def trajectory_distribution(self, xi, u, t):
        pass

    def get_seq(self, xi0, return_target=False):
        xis = [xi0]
        us = [-self._K[0].dot(xi0) + self._Kv[0].dot(self._v[0])]

        ds = []

        for t in range(1, self.horizon - 1):
            A = self.get_A(t)
            B = self.get_B(t)
            xis += [A.dot(xis[-1]) + B.dot(us[-1])]

            if return_target:
                d = np.linalg.inv(self._S[t].dot(A)).dot(self._v[t + 1])
                ds += [d]

                us += [self._K[t].dot(d - xis[-1])]
            else:
                us += [-self._K[t].dot(xis[-1]) + self._Kv[t].dot(self._v[t + 1])]

        if return_target:
            return np.array(xis), np.array(us), np.array(ds)
        else:
            return np.array(xis), np.array(us)

    def make_rollout_samples(self, x0):
        T = self.horizon

        xs = [None for i in range(T)]
        us = [None for i in range(T - 1)]

        xs[0] = x0
        next_xs = x0
        n = x0.shape[0]

        for i in range(T - 1):
            B = self.get_B(i)
            A = self.get_A(i)
            loc = self.get_command(next_xs, i)
            cov = self.Qc[i]
            eps = np.random.normal(size=(n, self.u_dim))
            next_us = loc + np.einsum('ij,aj->ai ', np.linalg.cholesky(cov), eps)
            next_xs = np.einsum('ij,aj->ai', A, next_xs) + np.einsum('ij,aj->ai', B, next_us)
            xs[i + 1] = next_xs
            us[i] = next_us
        return np.transpose(np.stack(xs), (1, 0, 2)), np.transpose(np.stack(us), (1, 0, 2))

    def make_rollout(self, x0):
        T = self.horizon

        xs = [None for i in range(T)]
        us = [None for i in range(T - 1)]

        xs[0] = x0

        for i in range(T - 1):
            B = self.get_B(i)
            A = self.get_A(i)

            next_us = self.get_command(xs[i], i)
            next_xs = np.einsum('ij,aj->ai', A, xs[i]) + np.einsum('ij,aj->ai', B, next_us)
            xs[i + 1] = next_xs
            us[i] = next_us
        return np.transpose(np.stack(xs), (1, 0, 2)), np.transpose(np.stack(us), (1, 0, 2))

    def rollout_policy(self, dist_policy, x0):
        """
        Rollout of the stochastic policy.
        :param dist_policy: A policy distribution which takes x and t as input.
        :param x0: initial state
        :return:
        """
        T = self.horizon

        xs = [None for i in range(T)]
        us = [None for i in range(T - 1)]
        xs[0] = x0
        for i in range(T - 1):
            B = self.get_B(i)
            A = self.get_A(i)
            next_us = dist_policy(xs[i], i).sample()
            next_xs = np.einsum('ij,aj->ai', A, xs[i]) + np.einsum('ij,aj->ai', B, next_us)
            xs[i + 1] = next_xs
            us[i] = next_us
        return np.transpose(np.stack(xs), (1, 0, 2)), np.transpose(np.stack(us), (1, 0, 2))


class GMMLQR(LQR):
    """
    LQR with a GMM cost on the state, approximation to be checked
    """

    def __init__(self, *args, **kwargs):
        self._full_gmm_xi = None
        LQR.__init__(self, *args, **kwargs)

    @property
    def full_gmm_xi(self):
        """
        Distribution of state
        :return:
        """
        return self._full_gmm_xi

    @full_gmm_xi.setter
    def full_gmm_xi(self, value):
        """
        :param value 		[pbd.GMM] or [(pbd.GMM, list)]
        """
        self._full_gmm_xi = value

    def ricatti(self, x0, n_best=None):
        costs = []

        if isinstance(self._full_gmm_xi, pbd.MTMM):
            full_gmm = self.full_gmm_xi.get_matching_gmm()
        else:
            full_gmm = self.full_gmm_xi

        if n_best is not None:
            log_prob_components = self.full_gmm_xi.log_prob_components(x0)
            a = np.sort(log_prob_components, axis=0)[-n_best - 1][0]

        for i in range(self.full_gmm_xi.nb_states):
            if n_best is not None and log_prob_components[i] < a:
                costs += [-np.inf]
            else:
                self.gmm_xi = full_gmm, [i for j in range(self.horizon)]
                LQR.ricatti(self)
                xis, us = self.get_seq(x0)
                costs += [np.sum(self.gmm_u.log_prob(us) + self.full_gmm_xi.log_prob(xis))]

        max_lqr = np.argmax(costs)
        self.gmm_xi = full_gmm, [max_lqr for j in range(self.horizon)]
        LQR.ricatti(self)


class PoGLQR(LQR):
    """
    Implementation of LQR with Product of Gaussian as described in

        http://calinon.ch/papers/Calinon - HFR2016.pdf

    """

    def __init__(self, A=None, B=None, nb_dim=2, dt=0.01, horizon=50):
        self._horizon = horizon
        self.A = A
        self.B = B
        self.nb_dim = nb_dim
        self.dt = dt

        self._s_xi, self._s_u = None, None
        self._x0 = None

        self._mvn_xi, self._mvn_u = None, None
        self._mvn_sol_xi, self._mvn_sol_u = None, None

        self._seq_xi, self._seq_u = None, None

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        self.reset_params()  # reset params
        self._A = value

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, value):
        self.reset_params()  # reset params
        self._B = value

    @property
    def mvn_u_dim(self):
        """
        Number of dimension of input sequence lifted form
        :return:
        """
        if self.B is not None:
            return self.B.shape[1] * self.horizon
        else:
            return self.nb_dim * self.horizon

    @property
    def mvn_xi_dim(self):

        """
        Number of dimension of state sequence lifted form
        :return:
        """
        if self.A is not None:
            return self.A.shape[0] * self.horizon
        else:
            return self.nb_dim * self.horizon * 2

    @property
    def mvn_sol_u(self):
        """
        Distribution of control input after solving LQR
        :return:
        """
        assert self.x0 is not None, "Please specify a starting state"
        assert self.mvn_xi is not None, "Please specify a target distribution"
        assert self.mvn_u is not None, "Please specify a control input distribution"

        if self._mvn_sol_u is None:
            self._mvn_sol_u = self.mvn_xi.inv_trans_s(
                self.s_u, self.s_xi.dot(self.x0)) % self.mvn_u

        return self._mvn_sol_u

    @property
    def seq_xi(self):
        if self._seq_xi is None:
            self._seq_xi = self.mvn_sol_xi.mu.reshape(self.horizon, self.xi_dim)

        return self._seq_xi

    @property
    def seq_u(self):
        if self._seq_u is None:
            self._seq_u = self.mvn_sol_u.mu.reshape(self.horizon, self.u_dim)

        return self._seq_u

    @property
    def mvn_sol_xi(self):
        """
        Distribution of state after solving LQR
        :return:
        """
        if self._mvn_sol_xi is None:
            # MU: self._mvn_sol_xi.mu -> self.s_u * mvn_sol_u.mu + self.s_xi.dot(self.x0)
            # it means: zeta = S_u * u + S_zeta * zeta_0
            # SIGMA: self._mvn_sol_xi.sigma -> self.s_u * mvn_sol_u.sigma * self.s_u^T
            # it means: Sigma_zeta = S_u * Sigma_u * S_u^T
            # Calinon, Stochastic learning and control in multiple coordinate systems
            self._mvn_sol_xi = self.mvn_sol_u.transform(
                self.s_u, self.s_xi.dot(self.x0))

        return self._mvn_sol_xi

    @property
    def mvn_xi(self):
        """
        Distribution of state
        :return:
        """
        return self._mvn_xi

    @mvn_xi.setter
    def mvn_xi(self, value):
        """
        :param value 		[float] or [pbd.MVN]
        """
        # resetting solution
        self._mvn_sol_xi = None
        self._mvn_sol_u = None
        self._seq_u = None
        self._seq_xi = None

        self._mvn_xi = value

    @property
    def mvn_u(self):
        """
        Distribution of control input
        :return:
        """
        return self._mvn_u

    @mvn_u.setter
    def mvn_u(self, value):
        """
        :param value 		[float] or [pbd.MVN]
        """
        # resetting solution
        self._mvn_sol_xi = None
        self._mvn_sol_u = None
        self._seq_u = None
        self._seq_xi = None

        if isinstance(value, pbd.MVN):
            self._mvn_u = value
        else:
            self._mvn_u = pbd.MVN(
                mu=np.zeros(self.mvn_u_dim), lmbda=10 ** value * np.eye(self.mvn_u_dim))

    @property
    def xis(self):
        return self.mvn_sol_xi.mu.reshape(self.horizon, self.xi_dim / self.horizon)

    @property
    def k(self):
        # return self.mvn_sol_u.sigma.dot(self.s_u.T.dot(self.mvn_xi.lmbda)).dot(self.s_xi).reshape(
        return self.mvn_sol_u.sigma.dot(self.s_u.T.dot(self.mvn_xi.lmbda)).dot(self.s_xi).reshape(
            (self.horizon, self.mvn_u_dim / self.horizon, self.mvn_xi_dim / self.horizon))

    @property
    def s_u(self):
        if self._s_u is None:
            # Calinon, Stochastic learning and control in multiple coordinate systems, Appendix II
            self._s_xi, self._s_u = lifted_transfer_matrix(self.A, self.B,
                                                           horizon=self.horizon, dt=self.dt, nb_dim=self.nb_dim)
        return self._s_u

    @property
    def s_xi(self):
        if self._s_xi is None:
            # Calinon, Stochastic learning and control in multiple coordinate systems, Appendix II
            self._s_xi, self._s_u = lifted_transfer_matrix(self.A, self.B,
                                                           horizon=self.horizon, dt=self.dt, nb_dim=self.nb_dim)

        return self._s_xi

    def reset_params(self):
        # reset everything
        self._s_xi, self._s_u = None, None
        self._x0 = None
        # self._mvn_xi, self._mvn_u = None, None
        self._mvn_sol_xi, self._mvn_sol_u = None, None
        self._seq_xi, self._seq_u = None, None

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, value):
        self.reset_params()

        self._horizon = value


class PoGLQRBi(LQR):
    """
    Implementation of LQR with Product of Gaussian as described in

        http://calinon.ch/papers/Calinon - HFR2016.pdf

    """

    def __init__(self, A=None, B=None, nb_dim=2, dt=0.01, horizon=50):
        # xi => zeta, x0 => zeta_1
        self._horizon = horizon  # T
        self.A = A
        self.B = B
        self.nb_dim = nb_dim  # D
        self.dt = dt

        self._s_xi, self._s_U = None, None
        self._x0_l, self._x0_r, self._x0_c = None, None, None

        self._mvn_xi_l, self._mvn_xi_r, self.mvn_xi_c, self._mvn_u = None, None, None, None
        self._mvn_sol_xi_l, self._mvn_sol_xi_r, self._mvn_sol_U = None, None, None

        self._seq_xi, self._seq_U = None, None

        self._C, self._C_l, self._C_r = None, None, None  # Coordination matrix

    # <editor-fold desc="Setter A, B, C, horizon, x0_l, x0_r, x0_c">
    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        self.reset_params()  # reset params
        self._A = value

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, value):
        self.reset_params()  # reset params
        self._B = value

    @property
    def C(self):
        self._C = np.hstack((np.eye(self.mvn_U_dim), -1 * np.eye(self.mvn_U_dim)))  # DT * DTK (100, 200)
        return self._C

    @property
    def C_l(self):
        self._C_l = np.hstack((np.eye(self.mvn_U_dim), 0 * np.eye(self.mvn_U_dim)))
        return self._C_l

    @property
    def C_r(self):
        self._C_r = np.hstack((0 * np.eye(self.mvn_U_dim), np.eye(self.mvn_U_dim)))
        return self._C_r

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, value):
        self.reset_params()

        self._horizon = value

    @property
    def x0_l(self):
        return self._x0_l

    @x0_l.setter
    def x0_l(self, value):
        # resetting solution
        self._mvn_sol_xi = None
        self._mvn_sol_u = None

        self._x0_l = value

    @property
    def x0_r(self):
        return self._x0_r

    @x0_r.setter
    def x0_r(self, value):
        # resetting solution
        self._mvn_sol_xi = None
        self._mvn_sol_u = None

        self._x0_r = value

    @property
    def x0_c(self):
        return self._x0_c

    @x0_c.setter
    def x0_c(self, value):
        # resetting solution
        self._mvn_sol_xi = None
        self._mvn_sol_u = None

        self._x0_c = value

    #
    # @x0_c.setter
    # def x0_c(self, value):
    #     # resetting solution
    #     self._mvn_sol_xi = None
    #     self._mvn_sol_u = None
    #
    #     self._x0_c = value
    # </editor-fold>

    # <editor-fold desc="U and xi dim">
    @property
    def mvn_U_dim(self):
        """
        Number of dimension of input sequence lifted form
        :return:
        """
        if self.B is not None:
            return self.B.shape[1] * self.horizon
        else:
            return self.nb_dim * self.horizon

    @property
    def mvn_xi_dim(self):

        """
        Number of dimension of state sequence lifted form
        :return:
        """
        if self.A is not None:
            return self.A.shape[0] * self.horizon
        else:
            return self.nb_dim * self.horizon * 2

    # </editor-fold>

    @property
    def mvn_sol_U(self):
        """
        Distribution of control input after solving LQR
        :return:
        """
        assert self.x0_l is not None, "Please specify a starting state"
        assert self.x0_r is not None, "Please specify a starting state"
        assert self.mvn_xi_l is not None, "Please specify a target distribution"
        assert self.mvn_xi_r is not None, "Please specify a target distribution"
        assert self.mvn_xi_c is not None, "Please specify a target distribution"
        assert self.mvn_u is not None, "Please specify a control input distribution"

        if self._mvn_sol_U is None:
            # self._mvn_sol_U = self.mvn_xi.inv_trans_s(self.s_U, self.s_xi.dot(self.x0)) % self.mvn_u

            a = self.mvn_xi_l.inv_trans_sC(self.s_U, self.s_xi.dot(self.x0_l), self.C_l)
            b = self.mvn_xi_r.inv_trans_sC(self.s_U, self.s_xi.dot(self.x0_r), self.C_r)
            c = self.mvn_xi_c.inv_trans_sC(self.s_U, self.s_xi.dot(self.x0_c), self.C)
            d = self.mvn_u.inv_trans_sR(self.s_U, self.C_l)
            e = self.mvn_u.inv_trans_sR(self.s_U, self.C_r)
            self._mvn_sol_U = self.get_sigma_mu(
                a // d // b // e // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c
                // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c
                // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c
                // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c
                // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c // c)

            # self._mvn_sol_U = self.get_sigma_mu(
            #     self.mvn_xi_l.inv_trans_sC(self.s_U, self.s_xi.dot(self.x0_l), self.C_l) % \
            #     self.mvn_xi_r.inv_trans_sC(self.s_U, self.s_xi.dot(self.x0_r), self.C_r) % \
            #     self.mvn_xi_c.inv_trans_sC(self.s_U, self.s_xi.dot(self.x0_c), self.C) \
            #     % self.mvn_u.inv_trans_sR(self.C))

        return self._mvn_sol_U

    def get_sigma_mu(self, prod):
        prod.sigma = np.linalg.pinv(prod.lmbda)
        prod.mu = prod.sigma.dot(prod.lmbda_mu)
        # sigma_U * \SUM s_U^T * Q_s * s_U * [C]^-1 (mu_U - s_xi*x0) + 0
        return prod

    @property
    def mvn_sol_xi(self):
        """
        Distribution of state after solving LQR
        :return:
        """
        if self._mvn_sol_xi is None:
            # MU: self._mvn_sol_xi.mu -> self.s_u * mvn_sol_u.mu + self.s_xi.dot(self.x0)
            # it means: zeta = S_u * u + S_zeta * zeta_0
            # SIGMA: self._mvn_sol_xi.sigma -> self.s_u * mvn_sol_u.sigma * self.s_u^T
            # it means: Sigma_zeta = S_u * Sigma_u * S_u^T
            # Calinon, Stochastic learning and control in multiple coordinate systems
            mvn_sol_U = self.mvn_sol_U
            mvn_sol_u_l = pbd.MVN(
                mu=mvn_sol_U.mu[:self.mvn_U_dim], sigma=mvn_sol_U.sigma[:self.mvn_U_dim, :self.mvn_U_dim])
            mvn_sol_u_r = pbd.MVN(
                mu=mvn_sol_U.mu[self.mvn_U_dim:], sigma=mvn_sol_U.sigma[self.mvn_U_dim:, self.mvn_U_dim:])

            self._mvn_sol_xi_l = mvn_sol_u_l.transform(self.s_U, self.s_xi.dot(self.x0_l))
            self._mvn_sol_xi_r = mvn_sol_u_r.transform(self.s_U, self.s_xi.dot(self.x0_r))
            # self._mvn_sol_xi_l = None
        return self._mvn_sol_xi_l, self._mvn_sol_xi_r

    @property
    def seq_xi(self):
        if self._seq_xi is None:
            mvn_sol_xi_l, mvn_sol_xi_r = self.mvn_sol_xi
            self._seq_xi_l = mvn_sol_xi_l.mu.reshape(self.horizon, self.xi_dim)
            self._seq_xi_r = mvn_sol_xi_r.mu.reshape(self.horizon, self.xi_dim)
            # self._seq_xi_l = None
        return self._seq_xi_l, self._seq_xi_r

    @property
    def seq_U(self):
        if self._seq_U is None:
            self._seq_U = self.mvn_sol_U.mu.reshape(self.horizon, self.U_dim)

        return self._seq_U

    # <editor-fold desc="Setter of mvn_xi and mvn_u">
    @property
    def mvn_xi_l(self):
        """
        Distribution of state
        :return:
        """
        return self._mvn_xi_l

    @mvn_xi_l.setter
    def mvn_xi_l(self, value):
        """
        :param value 		[float] or [pbd.MVN]
        """
        # resetting solution
        self._mvn_sol_xi = None
        self._mvn_sol_U = None
        self._seq_U = None
        self._seq_xi = None

        self._mvn_xi_l = value

    @property
    def mvn_xi_r(self):
        """
        Distribution of state
        :return:
        """
        return self._mvn_xi_r

    @mvn_xi_r.setter
    def mvn_xi_r(self, value):
        """
        :param value 		[float] or [pbd.MVN]
        """
        # resetting solution
        self._mvn_sol_xi = None
        self._mvn_sol_U = None
        self._seq_U = None
        self._seq_xi = None

        self._mvn_xi_r = value

    @property
    def mvn_xi_c(self):
        """
        Distribution of state
        :return:
        """
        return self._mvn_xi_c

    @mvn_xi_c.setter
    def mvn_xi_c(self, value):
        """
        :param value 		[float] or [pbd.MVN]
        """
        # resetting solution
        self._mvn_sol_xi = None
        self._mvn_sol_U = None
        self._seq_U = None
        self._seq_xi = None

        self._mvn_xi_c = value

    @property
    def mvn_u(self):
        """
        Distribution of control input
        :return:
        """
        return self._mvn_u

    @mvn_u.setter
    def mvn_u(self, value):
        """
        :param value 		[float] or [pbd.MVN]
        """
        # resetting solution
        self._mvn_sol_xi = None
        self._mvn_sol_U = None
        self._seq_U = None
        self._seq_xi = None

        if isinstance(value, pbd.MVN):
            self._mvn_u = value
        else:
            self._mvn_u = pbd.MVN(
                mu=np.zeros(self.mvn_U_dim), lmbda=10 ** value * np.eye(self.mvn_U_dim))

    # </editor-fold>

    @property
    def xis(self):
        return self.mvn_sol_xi.mu.reshape(self.horizon, self.xi_dim / self.horizon)

    @property
    def k(self):
        # return self.mvn_sol_u.sigma.dot(self.s_u.T.dot(self.mvn_xi.lmbda)).dot(self.s_xi).reshape(
        return self.mvn_sol_U.sigma.dot(self.s_U.T.dot(self.mvn_xi.lmbda)).dot(self.s_xi).reshape(
            (self.horizon, self.mvn_U_dim / self.horizon, self.mvn_xi_dim / self.horizon))

    @property
    def s_U(self):
        if self._s_U is None:
            # Calinon, Stochastic learning and control in multiple coordinate systems, Appendix II
            self._s_xi, self._s_U = lifted_transfer_matrix(self.A, self.B,
                                                           horizon=self.horizon, dt=self.dt, nb_dim=self.nb_dim)
        return self._s_U

    @property
    def s_xi(self):
        if self._s_xi is None:
            # Calinon, Stochastic learning and control in multiple coordinate systems, Appendix II
            self._s_xi, self._s_U = lifted_transfer_matrix(self.A, self.B,
                                                           horizon=self.horizon, dt=self.dt, nb_dim=self.nb_dim)
        return self._s_xi

    @property
    def U_dim(self):
        """
        Number of dimension of input
        :return:
        """
        if self.B is not None:
            return self.B.shape[1]
        else:
            return self.nb_dim

    def reset_params(self):
        # reset everything
        self._s_xi, self._s_U = None, None
        self._x0 = None
        # self._mvn_xi, self._mvn_u = None, None
        self._mvn_sol_xi, self._mvn_sol_U = None, None
        self._seq_xi, self._seq_U = None, None
