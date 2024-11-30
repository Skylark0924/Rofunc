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

from pbdlib.model import *

from rofunc.learning.RofuncML.gmm import GMM
from rofunc.utils.logger.beauty_logger import beauty_print


class HMM(GMM):
    def __init__(self, nb_states, nb_dim=2):
        GMM.__init__(self, nb_states, nb_dim)

        self._trans = None
        self._init_priors = None

    @property
    def init_priors(self):
        if self._init_priors is None:
            beauty_print("HMM init priors not defined, initializing to uniform", type="warning")
            self._init_priors = np.ones(self.nb_states) / self.nb_states

        return self._init_priors

    @init_priors.setter
    def init_priors(self, value):
        self._init_priors = value

    @property
    def trans(self):
        if self._trans is None:
            beauty_print("HMM transition matrix not defined, initializing to uniform", type="warning")
            self._trans = np.ones((self.nb_states, self.nb_states)) / self.nb_states
        return self._trans

    @trans.setter
    def trans(self, value):
        self._trans = value

    @property
    def Trans(self):
        return self.trans

    @Trans.setter
    def Trans(self, value):
        self.trans = value

    def make_finish_state(self, demos, dep_mask=None):
        self.has_finish_state = True
        self.nb_states += 1

        data = np.concatenate([d[-3:] for d in demos])

        mu = np.mean(data, axis=0)

        # Update covariances
        if data.shape[0] > 1:
            sigma = np.einsum('ai,aj->ij', data - mu, data - mu) / (data.shape[0] - 1) + self.reg
        else:
            sigma = self.reg

        # if cov_type == 'diag':
        # 	self.sigma *= np.eye(self.nb_dim)

        if dep_mask is not None:
            sigma *= dep_mask

        self.mu = np.concatenate([self.mu, mu[None]], axis=0)
        self.sigma = np.concatenate([self.sigma, sigma[None]], axis=0)
        self.init_priors = np.concatenate([self.init_priors, np.zeros(1)], axis=0)
        self.priors = np.concatenate([self.priors, np.zeros(1)], axis=0)
        pass

    def viterbi(self, demo, reg=True):
        """
        Compute most likely sequence of state given observations

        :param demo: 	[np.array([nb_timestep, nb_dim])]
        :return:
        """

        nb_data, dim = demo.shape if isinstance(demo, np.ndarray) else demo['x'].shape

        logB = np.zeros((self.nb_states, nb_data))
        logDELTA = np.zeros((self.nb_states, nb_data))
        PSI = np.zeros((self.nb_states, nb_data)).astype(int)

        _, logB = self.obs_likelihood(demo)

        # forward pass
        logDELTA[:, 0] = np.log(self.init_priors + realmin * reg) + logB[:, 0]

        for t in range(1, nb_data):
            for i in range(self.nb_states):
                # get index of maximum value : most probables
                PSI[i, t] = np.argmax(logDELTA[:, t - 1] + np.log(self.Trans[:, i] + realmin * reg))
                logDELTA[i, t] = np.max(logDELTA[:, t - 1] + np.log(self.Trans[:, i] + realmin * reg)) + logB[i, t]

        assert not np.any(np.isnan(logDELTA)), "Nan values"

        # backtracking
        q = [0 for i in range(nb_data)]
        q[-1] = np.argmax(logDELTA[:, -1])
        for t in range(nb_data - 2, -1, -1):
            q[t] = PSI[q[t + 1], t + 1]

        return q

    def split_kbins(self, demos):
        t_sep = []
        t_resp = []

        for demo in demos:
            t_sep += [map(int, np.round(
                np.linspace(0, demo.shape[0], self.nb_states + 1)))]

            resp = np.zeros((demo.shape[0], self.nb_states))

            # print t_sep
            for i in range(self.nb_states):
                resp[t_sep[-1][i]:t_sep[-1][i + 1], i] = 1.0
            # print resp
            t_resp += [resp]

        return np.concatenate(t_resp)

    def obs_likelihood(self, demo=None, dep=None, marginal=None, sample_size=200, demo_idx=None):
        sample_size = demo.shape[0]  # 50
        # emission probabilities
        B = np.ones((self.nb_states, sample_size))  # (4, 50)

        if marginal != []:
            for i in range(self.nb_states):
                mu, sigma = (self.mu, self.sigma)  # mu: (4, 8), sigma: (4, 8, 8)

                if marginal is not None:
                    mu, sigma = self.get_marginal(marginal)

                if dep is None:
                    B[i, :] = multi_variate_normal(demo, mu[i], sigma[i], log=True)

                else:  # block diagonal computation
                    B[i, :] = 0.
                    for d in dep:
                        if isinstance(d, list):
                            dGrid = np.ix_([i], d, d)
                            B[[i], :] += multi_variate_normal(demo[:, d], mu[i, d],
                                                              sigma[dGrid][0], log=True)
                        elif isinstance(d, slice):
                            B[[i], :] += multi_variate_normal(demo[:, d], mu[i, d],
                                                              sigma[i, d, d], log=True)

        return np.exp(B), B

    def online_forward_message(self, x, marginal=None, reset=False):
        """

        :param x:
        :param marginal: slice
        :param reset:
        :return:
        """
        if (not hasattr(self, '_marginal_tmp') or reset) and marginal is not None:
            self._marginal_tmp = self.marginal_model(marginal)

        if marginal is not None:
            B, _ = self._marginal_tmp.obs_likelihood(x[None])
        else:
            B, _ = self.obs_likelihood(x[None])

        if not hasattr(self, '_alpha_tmp') or reset:
            self._alpha_tmp = self.init_priors * B[:, 0]
        else:
            self._alpha_tmp = self._alpha_tmp.dot(self.Trans) * B[:, 0]

        self._alpha_tmp /= np.sum(self._alpha_tmp, keepdims=True)

        return self._alpha_tmp

    def compute_messages(self, demo=None, dep=None, table=None, marginal=None, sample_size=200, demo_idx=None):
        """

        :param demo: 	[np.array([nb_timestep, nb_dim])]
        :param dep: 	[A x [B x [int]]] A list of list of dimensions
            Each list of dimensions indicates a dependence of variables in the covariance matrix
            E.g. [[0],[1],[2]] indicates a diagonal covariance matrix
            E.g. [[0, 1], [2]] indicates a full covariance matrix between [0, 1] and no
            covariance with dim [2]
        :param table: 	np.array([nb_states, nb_demos]) - composed of 0 and 1
            A mask that avoid some demos to be assigned to some states
        :param marginal: [slice(dim_start, dim_end)] or []
            If not None, compute messages with marginals probabilities
            If [] compute messages without observations, use size
            (can be used for time-series regression)
        :return:
        """
        if isinstance(demo, np.ndarray):
            sample_size = demo.shape[0]
        elif isinstance(demo, dict):
            sample_size = demo['x'].shape[0]
        B, _ = self.obs_likelihood(demo, dep, marginal, sample_size)
        # if table is not None:
        # 	B *= table[:, [n]]
        self._B = B

        # forward variable alpha (rescaled)
        alpha = np.zeros((self.nb_states, sample_size))
        alpha[:, 0] = self.init_priors * B[:, 0]

        c = np.zeros(sample_size)
        c[0] = 1.0 / np.sum(alpha[:, 0] + realmin)
        alpha[:, 0] = alpha[:, 0] * c[0]

        for t in range(1, sample_size):
            alpha[:, t] = alpha[:, t - 1].dot(self.Trans) * B[:, t]
            # Scaling to avoid underflow issues
            c[t] = 1.0 / np.sum(alpha[:, t] + realmin)
            alpha[:, t] = alpha[:, t] * c[t]

        # backward variable beta (rescaled)
        beta = np.zeros((self.nb_states, sample_size))
        beta[:, -1] = np.ones(self.nb_states) * c[-1]  # Rescaling
        for t in range(sample_size - 2, -1, -1):
            beta[:, t] = np.dot(self.Trans, beta[:, t + 1] * B[:, t + 1])
            beta[:, t] = np.minimum(beta[:, t] * c[t], realmax)

        # Smooth node marginals, gamma
        gamma = (alpha * beta) / np.tile(np.sum(alpha * beta, axis=0) + realmin,
                                         (self.nb_states, 1))

        # Smooth edge marginals. zeta (fast version, considers the scaling factor)
        zeta = np.zeros((self.nb_states, self.nb_states, sample_size - 1))

        for i in range(self.nb_states):
            for j in range(self.nb_states):
                zeta[i, j, :] = self.Trans[i, j] * alpha[i, 0:-1] * B[j, 1:] * beta[
                                                                               j,
                                                                               1:]

        return alpha, beta, gamma, zeta, c

    def init_params_random(self, data, left_to_right=False, self_trans=0.9):
        """

        :param data:
        :param left_to_right:  	if True, init with left to right. All observations pre_trained_models
            will be the same, and transition matrix will be set to l_t_r
        :type left_to_right: 	bool
        :param self_trans:		if left_to_right, self transition value to fill
        :type self_trans:		float
        :return:
        """
        mu = np.mean(data, axis=0)
        sigma = np.cov(data.T)
        if sigma.ndim == 0:
            sigma = np.ones((1, 1)) * sigma

        if left_to_right:
            self.mu = np.array([mu for i in range(self.nb_states)])
        else:
            self.mu = np.array([np.random.multivariate_normal(mu * 1, sigma)
                                for i in range(self.nb_states)])

        self.sigma = np.array([sigma + self.reg for i in range(self.nb_states)])
        self.priors = np.ones(self.nb_states) / self.nb_states

        if left_to_right:
            self.Trans = np.zeros((self.nb_states, self.nb_states))
            for i in range(self.nb_states):
                if i < self.nb_states - 1:
                    self.Trans[i, i] = self_trans
                    self.Trans[i, i + 1] = 1. - self_trans
                else:
                    self.Trans[i, i] = 1.

            self.init_priors = np.zeros(self.nb_states) / self.nb_states
        else:
            self.Trans = np.ones((self.nb_states, self.nb_states)) * (1. - self_trans) / (self.nb_states - 1)
            # remove diagonal
            self.Trans *= (1. - np.eye(self.nb_states))
            self.Trans += self_trans * np.eye(self.nb_states)
            self.init_priors = np.ones(self.nb_states) / self.nb_states

    def gmm_init(self, data, **kwargs):
        if isinstance(data, list):
            data = np.concatenate(data, axis=0)
        GMM.em(self, data, **kwargs)

        self.init_priors = np.ones(self.nb_states) / self.nb_states
        self.Trans = np.ones((self.nb_states, self.nb_states)) / self.nb_states

    def init_loop(self, demos):
        self.Trans = 0.98 * np.eye(self.nb_states)
        for i in range(self.nb_states - 1):
            self.Trans[i, i + 1] = 0.02

        self.Trans[-1, 0] = 0.02

        data = np.concatenate(demos, axis=0)
        _mu = np.mean(data, axis=0)
        _cov = np.cov(data.T)

        self.mu = np.array([_mu for i in range(self.nb_states)])
        self.sigma = np.array([_cov for i in range(self.nb_states)])

        self.init_priors = np.array([1.] + [0. for i in range(self.nb_states - 1)])

    def em(self, demos, dep=None, reg=1e-8, table=None, end_cov=False, cov_type='full', dep_mask=None,
           reg_finish=None, left_to_right=False, nb_max_steps=40, loop=False, obs_fixed=False, trans_reg=None):
        """

        :param demos:	[list of np.array([nb_timestep, nb_dim])]
                or [lisf of dict({})]
        :param dep:		[A x [B x [int]]] A list of list of dimensions or slices
            Each list of dimensions indicates a dependence of variables in the covariance matrix
            !!! dimensions should not overlap eg : [[0], [0, 1]] should be [[0, 1]], [[0, 1], [1, 2]] should be [[0, 1, 2]]
            E.g. [[0],[1],[2]] indicates a diagonal covariance matrix
            E.g. [[0, 1], [2]] indicates a full covariance matrix between [0, 1] and no
            covariance with dim [2]
            E.g. [slice(0, 2), [2]] indicates a full covariance matrix between [0, 1] and no
            covariance with dim [2]
        :param reg:		[float] or list [nb_dim x float] for different regularization in different dimensions
            Regularization term used in M-step for covariance matrices
        :param table:		np.array([nb_states, nb_demos]) - composed of 0 and 1
            A mask that avoid some demos to be assigned to some states
        :param end_cov:	[bool]
            If True, compute covariance matrix without regularization after convergence
        :param cov_type: 	[string] in ['full', 'diag', 'spherical']
        :return:
        """

        if reg_finish is not None: end_cov = True

        nb_min_steps = 2  # min num iterations
        max_diff_ll = 1e-4  # max log-likelihood increase

        nb_samples = len(demos)
        data = np.concatenate(demos).T
        nb_data = data.shape[0]

        s = [{} for d in demos]
        # stored log-likelihood
        LL = np.zeros(nb_max_steps)

        if dep is not None:
            dep_mask = self.get_dep_mask(dep)

        self.reg = reg

        if self.mu is None or self.sigma is None:
            self.init_params_random(data.T, left_to_right=left_to_right)

        # create regularization matrix

        if left_to_right or loop:
            mask = np.eye(self.Trans.shape[0])
            for i in range(self.Trans.shape[0] - 1):
                mask[i, i + 1] = 1.
            if loop:
                mask[-1, 0] = 1.

        if dep_mask is not None:
            self.sigma *= dep_mask

        for it in range(nb_max_steps):

            for n, demo in enumerate(demos):
                s[n]['alpha'], s[n]['beta'], s[n]['gamma'], s[n]['zeta'], s[n]['c'] = HMM.compute_messages(self, demo,
                                                                                                           dep, table)

            # concatenate intermediary vars
            gamma = np.hstack([s[i]['gamma'] for i in range(nb_samples)])
            zeta = np.dstack([s[i]['zeta'] for i in range(nb_samples)])
            gamma_init = np.hstack([s[i]['gamma'][:, 0:1] for i in range(nb_samples)])
            gamma_trk = np.hstack([s[i]['gamma'][:, 0:-1] for i in range(nb_samples)])

            gamma2 = gamma / (np.sum(gamma, axis=1, keepdims=True) + realmin)

            # M-step
            if not obs_fixed:
                for i in range(self.nb_states):
                    # Update centers
                    self.mu[i] = np.einsum('a,ia->i', gamma2[i], data)

                    # Update covariances
                    Data_tmp = data - self.mu[i][:, None]
                    self.sigma[i] = np.einsum('ij,jk->ik',
                                              np.einsum('ij,j->ij', Data_tmp,
                                                        gamma2[i, :]), Data_tmp.T)
                    # Regularization
                    self.sigma[i] = self.sigma[i] + self.reg

                    if cov_type == 'diag':
                        self.sigma[i] *= np.eye(self.sigma.shape[1])

                if dep_mask is not None:
                    self.sigma *= dep_mask

            # Update initial state probablility vector
            self.init_priors = np.mean(gamma_init, axis=1)

            # Update transition probabilities
            self.Trans = np.sum(zeta, axis=2) / (np.sum(gamma_trk, axis=1) + realmin)

            if trans_reg is not None:
                self.Trans += trans_reg
                self.Trans /= np.sum(self.Trans, axis=1, keepdims=True)

            if left_to_right or loop:
                self.Trans *= mask
                self.Trans /= np.sum(self.Trans, axis=1, keepdims=True)

            # print self.Trans
            # Compute avarage log-likelihood using alpha scaling factors
            LL[it] = 0
            for n in range(nb_samples):
                LL[it] -= sum(np.log(s[n]['c']))
            LL[it] = LL[it] / nb_samples

            self._gammas = [s_['gamma'] for s_ in s]

            # Check for convergence
            if it > nb_min_steps and LL[it] - LL[it - 1] < max_diff_ll:
                print("EM converges")
                if end_cov:
                    for i in range(self.nb_states):
                        # recompute covariances without regularization
                        Data_tmp = data - self.mu[i][:, None]
                        self.sigma[i] = np.einsum('ij,jk->ik',
                                                  np.einsum('ij,j->ij', Data_tmp,
                                                            gamma2[i, :]), Data_tmp.T)
                        if reg_finish is not None:
                            self.reg = reg_finish
                            self.sigma += self.reg[None]

                    if cov_type == 'diag':
                        self.sigma[i] *= np.eye(self.sigma.shape[1])

                # print "EM converged after " + str(it) + " iterations"
                # print LL[it]

                if dep_mask is not None:
                    self.sigma *= dep_mask

                return True

        print("EM did not converge")
        return False

    def score(self, demos):
        """

        :param demos:	[list of np.array([nb_timestep, nb_dim])]
        :return:
        """
        ll = []
        for n, demo in enumerate(demos):
            _, _, _, _, c = HMM.compute_messages(self, demo)
            ll += [np.sum(np.log(c))]

        return ll

    def condition(self, data_in, dim_in, dim_out, h=None, return_gmm=False):
        if return_gmm:
            return super().condition(data_in, dim_in, dim_out, return_gmm=return_gmm)
        else:
            if dim_in == slice(0, 1):
                dim_in_msg = []
            else:
                dim_in_msg = dim_in
            a, _, _, _, _ = self.compute_messages(data_in, marginal=dim_in_msg)

            return super().condition(data_in, dim_in, dim_out, h=a)

    """
    To ensure compatibility
    """

    @property
    def Trans(self):
        return self.trans

    @Trans.setter
    def Trans(self, value):
        self.trans = value
