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
from pbdlib.mvn import MVN
from scipy.linalg import block_diag
from scipy.special import logsumexp
from termcolor import colored


class GMM(Model):
    def __init__(self, nb_states=1, nb_dim=None, init_zeros=False, mu=None, lmbda=None, sigma=None, priors=None,
                 log_priors=None):
        if mu is not None:
            nb_states = mu.shape[0]
            nb_dim = mu.shape[-1]

        super().__init__(nb_states, nb_dim)

        # flag to indicate that publishing was not init
        self.publish_init = False

        self._mu = mu
        self._lmbda = lmbda
        self._sigma = sigma
        self._priors = priors
        self._log_priors = log_priors

        if init_zeros:
            self.init_zeros()

    def get_matching_mvn(self, max=False, mass=None):
        if max:
            priors = (self.priors == np.max(self.priors)).astype(np.float32)
            priors /= np.sum(priors)
        elif mass is not None:
            prior_lim = np.sort(self.priors)[::-1][np.max(
                [0, np.argmin(np.cumsum(np.sort(self.priors)[::-1]) < mass)])]

            priors = (self.priors >= prior_lim) * self.priors
            priors /= np.sum(priors)
        else:
            priors = self.priors
        # print priors, self.priors

        mus, sigmas = self.moment_matching(priors)
        mvn = MVN(nb_dim=self.nb_dim, mu=mus, sigma=sigmas)

        return mvn

    def moment_matching(self, h):
        """
        Perform moment matching to approximate a mixture of Gaussian as a Gaussian
        :param h: 		np.array([nb_timesteps, nb_states])
            Activations of each states for different timesteps
        :return:
        """
        if h.ndim == 1:
            h = h[None]

        if self.mu.ndim == 2:
            mus = np.einsum('ak,ki->ai', h, self.mu)
            dmus = self.mu[None] - mus[:, None]  # nb_timesteps, nb_states, nb_dim
            sigmas = np.einsum('ak,kij->aij', h, self.sigma) + \
                     np.einsum('ak,akij->aij', h, np.einsum('aki,akj->akij', dmus, dmus))
        else:
            mus = np.einsum('ak,aki->ai', h, self.mu)
            dmus = self.mu - mus[:, None]  # nb_timesteps, nb_states, nb_dim
            sigmas = np.einsum('ak,akij->aij', h, self.sigma) + \
                     np.einsum('ak,akij->aij', h, np.einsum('aki,akj->akij', dmus, dmus))

        return mus, sigmas

    def __add__(self, other):
        if isinstance(other, MVN):
            gmm = GMM(nb_dim=self.nb_dim, nb_states=self.nb_states)

            gmm.priors = self.priors
            gmm.mu = self.mu + other.mu[None]
            gmm.sigma = self.sigma + other.sigma[None]

            return gmm

        else:
            raise NotImplementedError

    def __mul__(self, other):
        """
        Renormalized product of Gaussians, component by component

        :param other:
        :return:
        """
        if isinstance(other, MVN):
            gmm = GMM(nb_dim=self.nb_dim, nb_states=self.nb_states)
            gmm.mu = np.einsum('aij,aj->ai', self.lmbda, self.mu) + \
                     np.einsum('ij,j->i', other.lmbda, other.mu)[None]

            gmm.lmbda = self.lmbda + other.lmbda[None]
            gmm.mu = np.einsum('aij,aj->ai', gmm.sigma, gmm.mu)

            Z = np.linalg.slogdet(self.lmbda)[1] \
                + np.linalg.slogdet(other.lmbda)[1] \
                - 0.5 * np.linalg.slogdet(gmm.lmbda)[1] \
                - self.nb_dim / 2. * np.log(2 * np.pi) \
                + 0.5 * (np.einsum('ai,aj->a',
                                   np.einsum('ai,aij->aj', gmm.mu, gmm.lmbda), gmm.mu)
                         - np.einsum('ai,aj->a',
                                     np.einsum('ai,aij->aj', self.mu, self.lmbda), self.mu)
                         - np.sum(np.einsum('i,ij->j', other.mu, other.lmbda) * other.mu)
                         )
            gmm.priors = np.exp(Z) * self.priors
            gmm.priors /= np.sum(gmm.priors)

        else:
            # component wise
            gmm = GMM(nb_dim=self.nb_dim, nb_states=self.nb_states)
            gmm.priors = self.priors
            gmm.mu = np.einsum('aij,aj->ai', self.lmbda, self.mu) + \
                     np.einsum('aij,aj->ai', other.lmbda, other.mu)

            gmm.lmbda = self.lmbda + other.lmbda

            gmm.mu = np.einsum('aij,aj->ai', gmm.sigma, gmm.mu)

        return gmm

    def __mod__(self, other):
        """
        Renormalized product of Gaussians, component by component

        :param other:
        :return:
        """

        gmm = GMM(nb_dim=self.nb_dim, nb_states=self.nb_states * other.nb_states)
        # gmm.priors = self.priors
        gmm.mu = np.einsum('aij,aj->ai', self.lmbda, self.mu)[:, None] + \
                 np.einsum('aij,aj->ai', other.lmbda, other.mu)[None]

        gmm.lmbda = self.lmbda[:, None] + other.lmbda[None]

        gmm.sigma = np.linalg.inv(gmm.lmbda)

        gmm.mu = np.einsum('abij,abj->abi', gmm.sigma, gmm.mu)

        return gmm

    def marginal_model(self, dims):
        """
        Get a GMM of a slice of this GMM
        :param dims:
        :type dims: slice
        :return:
        """
        gmm = GMM(nb_dim=dims.stop - dims.start, nb_states=self.nb_states)
        gmm.priors = self.priors
        gmm.mu = self.mu[:, dims]
        gmm.sigma = self.sigma[:, dims, dims]

        return gmm

    def lintrans(self, A, b):
        """
        Linear transformation of a GMM

        :param A:		np.array(nb_dim, nb_dim)
        :param b: 		np.array(nb_dim)
        :return:
        """

        gmm = GMM(nb_dim=self.nb_dim, nb_states=self.nb_states)
        gmm.priors = self.priors
        gmm.mu = np.einsum('ij,aj->ai', A, self.mu) + b
        gmm.lmbda = np.einsum('aij,kj->aik',
                              np.einsum('ij,ajk->aik', A, self.lmbda), A)

        return gmm

    def lintrans_dyna(self, A, b):
        """
        Linear transformation of a GMM

        :param A:		np.array(nb_states, nb_dim, nb_dim)
        :param b: 		np.array(nb_states, nb_dim)
        :return:
        """

        gmm = GMM(nb_dim=self.nb_dim, nb_states=self.nb_states)
        gmm.priors = self.priors
        gmm.mu = np.einsum('aij,aj->ai', A, self.mu) + b
        gmm.lmbda = np.einsum('aij,akj->aik',
                              np.einsum('aij,ajk->aik', A, self.lmbda), A)

        return gmm

    def concatenate_gaussian(self, q, get_mvn=True, reg=None):
        """
        Get a concatenated-block-diagonal replication of the GMM with sequence of state
        given by q.

        :param q: 			[list of int]
        :param get_mvn: 	[bool]


        :return:
        """
        if reg is None:
            if not get_mvn:
                return np.concatenate([self.mu[i] for i in q]), block_diag(*[self.sigma[i] for i in q])
            else:
                mvn = MVN()
                mvn.mu = np.concatenate([self.mu[i] for i in q])
                mvn._sigma = block_diag(*[self.sigma[i] for i in q])
                mvn._lmbda = block_diag(*[self.lmbda[i] for i in q])

                return mvn
        else:
            if not get_mvn:
                return np.concatenate([self.mu[i] for i in q]), block_diag(
                    *[self.sigma[i] + reg for i in q])
            else:
                mvn = MVN()
                mvn.mu = np.concatenate([self.mu[i] for i in q])
                mvn._sigma = block_diag(*[self.sigma[i] + reg for i in q])
                mvn._lmbda = block_diag(*[np.linalg.inv(self.sigma[i] + reg) for i in q])

                return mvn

    def compute_resp(self, demo=None, dep=None, table=None, marginal=None, norm=True):
        sample_size = demo.shape[0]

        B = np.ones((self.nb_states, sample_size))

        if marginal != []:
            for i in range(self.nb_states):
                mu, sigma = (self.mu, self.sigma)

                if marginal is not None:
                    mu, sigma = self.get_marginal(marginal)

                if dep is None:
                    B[i, :] = multi_variate_normal(demo,
                                                   mu[i],
                                                   sigma[i], log=False)
                else:  # block diagonal computation
                    B[i, :] = 1.0
                    for d in dep:
                        dGrid = np.ix_([i], d, d)
                        B[[i], :] *= multi_variate_normal(demo, mu[i, d],
                                                          sigma[dGrid][:, :, 0], log=False)
        B *= self.priors[:, None]
        if norm:
            return B / np.sum(B, axis=0)
        else:
            return B

    def init_params_scikit(self, data, cov_type='full'):
        from sklearn.mixture import GaussianMixture
        gmm_init = GaussianMixture(self.nb_states, cov_type, n_init=5, init_params='random')
        gmm_init.fit(data)

        self.mu = gmm_init.means_
        if cov_type == 'diag':
            self.sigma = np.array([np.diag(gmm_init.covariances_[i]) for i in range(self.nb_states)])
        else:
            self.sigma = gmm_init.covariances_

        self.priors = gmm_init.weights_

        self.Trans = np.ones((self.nb_states, self.nb_states)) * 0.01

        self.init_priors = np.ones(self.nb_states) * 1. / self.nb_states

    def init_params_kmeans(self, data):
        from sklearn.cluster import KMeans
        km_init = KMeans(n_clusters=self.nb_states)
        km_init.fit(data)
        self.mu = km_init.cluster_centers_
        self.priors = np.ones(self.nb_states) / self.nb_states
        self.sigma = np.array([np.eye(self.nb_dim) for i in range(self.nb_states)])

        self.Trans = np.ones((self.nb_states, self.nb_states)) * 0.01

        self.init_priors = np.ones(self.nb_states) * 1. / self.nb_states

    def init_params_random(self, data):
        mu = np.mean(data, axis=0)
        sigma = np.dot((data - mu).T, (data - mu)) / \
                (data.shape[0] - 1)

        self.mu = np.array([np.random.multivariate_normal(mu, sigma)
                            for i in range(self.nb_states)])

        self.sigma = np.array([sigma + self.reg for i in range(self.nb_states)])

        self.priors = np.ones(self.nb_states) / self.nb_states

    def em(self, data, reg=1e-8, maxiter=100, minstepsize=1e-5, diag=False, reg_finish=False,
           kmeans_init=False, random_init=True, dep_mask=None, verbose=False, only_scikit=False,
           no_init=False):
        """

        :param data:	 		[np.array([nb_timesteps, nb_dim])]
        :param reg:				[list([nb_dim]) or float]
            Regulariazation for EM
        :param maxiter:
        :param minstepsize:
        :param diag:			[bool]
            Use diagonal covariance matrices
        :param reg_finish:		[np.array([nb_dim]) or float]
            Regulariazation for finish step
        :param kmeans_init:		[bool]
            Init components with k-means.
        :param random_init:		[bool]
            Init components randomely.
        :param dep_mask: 		[np.array([nb_dim, nb_dim])]
            Composed of 0 and 1. Mask given the dependencies in the covariance matrices
        :return:
        """
        if self.nb_dim is None:
            self.nb_dim = data.shape[-1]

        self.reg = reg

        nb_min_steps = 5  # min num iterations
        nb_max_steps = maxiter  # max iterations
        max_diff_ll = minstepsize  # max log-likelihood increase

        nb_samples = data.shape[0]

        if not no_init:
            if random_init and not only_scikit:
                self.init_params_random(data)
            elif kmeans_init and not only_scikit:
                self.init_params_kmeans(data)
            else:
                if diag:
                    self.init_params_scikit(data, 'diag')
                else:
                    self.init_params_scikit(data, 'full')

        if only_scikit: return
        data = data.T

        LL = np.zeros(nb_max_steps)
        for it in range(nb_max_steps):

            # E - step
            L = np.zeros((self.nb_states, nb_samples))
            L_log = np.zeros((self.nb_states, nb_samples))

            for i in range(self.nb_states):
                L_log[i, :] = np.log(self.priors[i]) + multi_variate_normal(data.T, self.mu[i],
                                                                            self.sigma[i], log=True)

            L = np.exp(L_log)
            GAMMA = L / np.sum(L, axis=0)
            GAMMA2 = GAMMA / np.sum(GAMMA, axis=1)[:, np.newaxis]

            # M-step
            self.mu = np.einsum('ac,ic->ai', GAMMA2,
                                data)  # a states, c sample, i dim

            dx = data[None, :] - self.mu[:, :, None]  # nb_dim, nb_states, nb_samples

            self.sigma = np.einsum('acj,aic->aij', np.einsum('aic,ac->aci', dx, GAMMA2),
                                   dx)  # a states, c sample, i-j dim

            self.sigma += self.reg

            if diag:
                self.sigma *= np.eye(self.nb_dim)

            if dep_mask is not None:
                self.sigma *= dep_mask

            # print self.Sigma[:,u :, i]

            # Update initial state probablility vector
            self.priors = np.mean(GAMMA, axis=1)

            LL[it] = np.mean(np.log(np.sum(L, axis=0)))
            # Check for convergence
            if it > nb_min_steps:
                if LL[it] - LL[it - 1] < max_diff_ll:
                    if reg_finish is not False:
                        self.sigma = np.einsum(
                            'acj,aic->aij', np.einsum('aic,ac->aci', dx, GAMMA2), dx) + reg_finish

                    if verbose:
                        print(colored('Converged after %d iterations: %.3e' % (it, LL[it]), 'red', 'on_white'))
                    return GAMMA
        if verbose:
            print(
                "GMM did not converge before reaching max iteration. Consider augmenting the number of max iterations.")
        return GAMMA

    def init_hmm_kbins(self, demos, dep=None, reg=1e-8, dep_mask=None):
        """
        Init HMM by splitting each demos in K bins along time. Each K states of the HMM will
        be initialized with one of the bin. It corresponds to a left-to-right HMM.

        :param demos:	[list of np.array([nb_timestep, nb_dim])]
        :param dep:
        :param reg:		[float]
        :return:
        """

        # delimit the cluster bins for first demonstration
        self.nb_dim = demos[0].shape[1]

        self.init_zeros()

        t_sep = []

        for demo in demos:
            t_sep += [list(map(int, np.round(np.linspace(0, demo.shape[0], self.nb_states + 1))))]

        # print t_sep
        for i in range(self.nb_states):
            data_tmp = np.empty((0, self.nb_dim))
            inds = []
            states_nb_data = 0  # number of datapoints assigned to state i

            # Get bins indices for each demonstration
            for n, demo in enumerate(demos):
                inds = range(t_sep[n][i], t_sep[n][i + 1])

                data_tmp = np.concatenate([data_tmp, demo[inds]], axis=0)
                states_nb_data += t_sep[n][i + 1] - t_sep[n][i]

            self.priors[i] = states_nb_data
            self.mu[i] = np.mean(data_tmp, axis=0)

            if dep_mask is not None:
                self.sigma *= dep_mask

            if dep is None:
                self.sigma[i] = np.cov(data_tmp.T) + np.eye(self.nb_dim) * reg
            else:
                for d in dep:
                    dGrid = np.ix_([i], d, d)
                    self.sigma[dGrid] = (np.cov(data_tmp[:, d].T) + np.eye(
                        len(d)) * reg)[:, :, np.newaxis]
        # print self.Sigma[:,:,i]

        # normalize priors
        self.priors = self.priors / np.sum(self.priors)

        # Hmm specific init
        self.Trans = np.ones((self.nb_states, self.nb_states)) * 0.01

        nb_data = np.mean([d.shape[0] for d in demos])

        for i in range(self.nb_states - 1):
            self.Trans[i, i] = 1.0 - float(self.nb_states) / nb_data
            self.Trans[i, i + 1] = float(self.nb_states) / nb_data

        self.Trans[-1, -1] = 1.0
        self.init_priors = np.ones(self.nb_states) * 1. / self.nb_states

    def add_trash_component(self, data, scale=2.):
        if isinstance(data, list):
            data = np.concatenate(data, axis=0)

        mu_new = np.mean(data, axis=0)
        sigma_new = scale ** 2 * np.cov(data.T)

        self.priors = np.concatenate([self.priors, 0.01 * np.ones(1)])
        self.priors /= np.sum(self.priors)
        self.mu = np.concatenate([self.mu, mu_new[None]], axis=0)
        self.sigma = np.concatenate([self.sigma, sigma_new[None]], axis=0)

    def mvn_pdf(self, x, reg=None):
        """

        :param x: 			np.array([nb_samples, nb_dim])
            samples
        :param mu: 			np.array([nb_states, nb_dim])
            mean vector
        :param sigma_chol: 	np.array([nb_states, nb_dim, nb_dim])
            cholesky decomposition of covariance matrices
        :param lmbda: 		np.array([nb_states, nb_dim, nb_dim])
            precision matrices
        :return: 			np.array([nb_states, nb_samples])
            log mvn
        """

        mu, lmbda_, sigma_chol_ = self.mu, self.lmbda, self.sigma_chol

        if x.ndim > 1 and mu.ndim == 2:
            dx = mu[None] - x[:, None]  # nb_timesteps, nb_states, nb_dim
            eins_idx = ('baj,baj->ba', 'ajk,baj->bak')
        elif x.ndim > 1 and mu.ndim == 3:
            dx = mu - x[:, None]  # nb_timesteps, nb_states, nb_dim
            eins_idx = ('baj,baj->ba', 'bajk,baj->bak')
        else:
            dx = mu - x
            eins_idx = ('aj,aj->a', 'ajk,aj->ak')

        if lmbda_.ndim == 4:
            cov_part = np.sum(np.log(sigma_chol_.diagonal(axis1=2, axis2=3)), axis=-1)

        else:
            cov_part = np.sum(np.log(sigma_chol_.diagonal(axis1=1, axis2=2)), axis=1)

        return -0.5 * np.einsum(eins_idx[0], dx, np.einsum(eins_idx[1], lmbda_, dx)) \
            - (mu.shape[-1] / 2.) * np.log(2 * np.pi) - cov_part

    def log_prob(self, x):

        return logsumexp(self.log_priors + self.mvn_pdf(x), -1)
