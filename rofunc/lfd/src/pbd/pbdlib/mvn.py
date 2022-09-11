import numpy as np

prec_min = 1e-15
import sys
from .utils.gaussian_utils import gaussian_conditioning
from .functions import mvn_pdf
from .functions import multi_variate_normal
import pbdlib as pbd
from scipy.linalg import sqrtm


class MVN(object):
    def __init__(self, mu=None, sigma=None, lmbda=None, lmbda_ns=None, sigma_cv=None, nb_dim=2):
        """
        Multivariate Normal Distribution


        :param mu:		np.array([nb_dim])
            Mean vector
        :param sigma: 	np.array([nb_dim, nb_dim])
            Covariance matrix
        :param lmbda: 	np.array([nb_dim, nb_dim])
            Precision matrix
        :param lmbda_ns:
        :param sigma_cv:
        """

        self._mu = mu
        self._sigma = sigma
        self._lmbda = lmbda
        self._sigma_chol = None
        self._eta = None
        self._C = None
        self._s_U = None
        #
        self.lmbda_ns = lmbda_ns
        self.sigma_cv = sigma_cv

        self._lmbdaT = None
        self._muT = None

        if mu is not None:
            self.nb_dim = mu.shape[0]
        elif sigma is not None:
            self.nb_dim = sigma.shape[0]
        elif lmbda is not None:
            self.nb_dim = lmbda.shape[0]
        else:
            self.nb_dim = nb_dim

    @property
    def eta(self):
        """
        Natural parameters eta = lambda.dot(mu)

        :return:
        """
        if self._eta is None:
            self._eta = self.lmbda.dot(self.mu)

        return self._eta

    @property
    def mu(self):
        if self._mu is None:
            self._mu = np.zeros(self.nb_dim)
        return self._mu

    @mu.setter
    def mu(self, value):
        self.nb_dim = value.shape[-1]
        self._mu = value
        self._eta = None

    @property
    def C(self):
        if self._C is None:
            self._C = np.zeros(self.nb_dim)
        return self._C

    @C.setter
    def C(self, value):
        self._C = value

    @property
    def s_U(self):
        if self._s_U is None:
            self._s_U = np.zeros(self.nb_dim)
        return self._s_U

    @s_U.setter
    def s_U(self, value):
        self._s_U = value

    @property
    def sigma(self):
        if self._sigma is None and not self._lmbda is None:
            try:
                self._sigma = np.linalg.inv(self._lmbda)  # 逆矩阵
                # except np.linalg.LinAlgError:
            except np.linalg.LinAlgError:
                self._sigma = np.linalg.inv(self._lmbda + prec_min * np.eye(self._lmbda.shape[0]))
            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise

        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self.nb_dim = value.shape[-1]
        self._lmbda = None
        self._sigma_chol = None
        self._sigma = value
        self._eta = None

    def plot(self, *args, **kwargs):
        pbd.plot_gaussian(self.mu, self.sigma, *args, **kwargs)

    @property
    def muT(self):
        """
        Returns muT-b
        :return:
        """
        if self._muT is not None:
            return self._muT
        else:
            return self.mu

    @property
    def lmbdaT(self):
        """
        Returns A^T.dot(lmbda)
        :return:
        """
        if self._lmbdaT is not None:
            return self._lmbdaT
        else:
            return self.lmbda

    @property
    def lmbda(self):
        if self._lmbda is None and not self._sigma is None:
            self._lmbda = np.linalg.inv(self._sigma)
        return self._lmbda

    @lmbda.setter
    def lmbda(self, value):
        self._sigma = None  # reset sigma
        self._sigma_chol = None
        self._lmbda = value
        self._eta = None

    @property
    def sigma_chol(self):
        if self.sigma is None:
            return None
        else:
            if self._sigma_chol is None:
                self._sigma_chol = np.linalg.cholesky(self.sigma)
            return self._sigma_chol

    def ml(self, data):
        self.mu = np.mean(data, axis=-1)
        self.sigma = np.cov(data.T)
        self.lmbda = np.linalg.inv(self.sigma)

    def log_prob(self, x, marginal=None, reg=None):
        """

        :param x:
        :param marginal:
        :type marginal: slice
        :return:
        """
        if marginal is not None:
            if self._batch:
                _mu = self.mu[..., marginal]
                _sigma = self.sigma[..., marginal, marginal]
            else:
                _mu = self.mu[marginal]
                _sigma = self.sigma[marginal, marginal]

            if reg is not None:
                _sigma += np.eye(marginal.stop - marginal.start) * reg
            return multi_variate_normal(x, _mu, _sigma)

        return multi_variate_normal(x, self.mu, self.sigma)

    def transform(self, A, b=None, dA=None, db=None):
        """
        The linear transform of a Gaussian
        if x ~ N(mu, Sigma)
        then Ax+b ~ N(A*mu+b, A*Sigma*A^T)
        """
        if b is None: b = np.zeros(A.shape[0])
        if dA is None:
            return type(self)(mu=A.dot(self.mu) + b, sigma=A.dot(self.sigma).dot(A.T))
        else:
            return self.transform_uncertainty(A, b, dA=None, db=None)

    def inv_transform(self, A, b):
        """

        :param A:		[np.array((nb_dim_expert, nb_dim_data))]
            Transformation under which the expert was seeing the data: A.dot(x)
        :param b: 		[np.array()]
        :return:
        """
        A_pinv = np.linalg.pinv(A)
        lmbda = A.T.dot(self.lmbda).dot(A)

        return type(self)(mu=A_pinv.dot(self.mu - b), lmbda=lmbda)

    def inv_trans_s(self, A, b):
        """

        :param A:
        :param b:
        :return:
        """
        mvn = type(self)(nb_dim=A.shape[1])
        mvn._muT = self.mu - b
        mvn._lmbdaT = A.T.dot(self.lmbda)
        mvn.lmbda = A.T.dot(self.lmbda).dot(A)

        return mvn

    def inv_trans_sC(self, A, b, C):
        """

        :param A:
        :param b:
        :return:
        """
        mvn = type(self)(nb_dim=A.shape[1])
        mvn._muT = self.mu - b
        mvn._lmbdaT = C.T.dot(A.T).dot(self.lmbda)  # \SUM C^T * s_U^T * Q_s
        mvn.lmbda = C.T.dot(A.T).dot(self.lmbda).dot(A).dot(C)  # \SUM C^T * s_U^T * Q_s * S_U * C
        mvn.lmbda_mu = C.T.dot(A.T).dot(self.lmbda).dot(self.mu - b)
        # mvn._lmbdaT = A.T.dot(self.lmbda)  # \SUM C^T * s_U^T * Q_s
        # mvn.lmbda = A.T.dot(self.lmbda).dot(A)  # \SUM C^T * s_U^T * Q_s * S_U * C
        # mvn.lmbda_mu = A.T.dot(self.lmbda).dot(self.mu - b)

        return mvn

    def inv_trans_sR(self, A, C):
        """

        :param A:
        :param b:
        :return:
        """
        mvn = type(self)(nb_dim=A.shape[1])
        mvn.lmbda = C.T.dot(self.lmbda).dot(C)  # \SUM C^T * s_U^T * Q_s * S_U * C
        mvn.lmbda_mu = C.T.dot(self.lmbda).dot(mvn.muT)

        return mvn

    def condition(self, data, dim_in, dim_out):
        mu, sigma = gaussian_conditioning(
            self.mu, self.sigma, data, dim_in, dim_out)

        if data.ndim == 1:
            conditional_mvn = type(self)(mu=mu[0], sigma=sigma[0])
        else:
            conditional_mvn = pbd.GMM()
            conditional_mvn.mu, conditional_mvn.sigma = mu, sigma

        return conditional_mvn

    def __add__(self, other):
        """
        Distribution of the sum of two random variables normally distributed

        :param other:
        :return:
        """
        assert self.mu.shape == other.mu.shape, "MVNs should be of same dimensions"

        mvn_sum = type(self)()

        mvn_sum.mu = self.mu + other.mu
        mvn_sum.sigma = self.sigma + other.sigma

        return mvn_sum

    def __mul__(self, other):
        """
        Standart product of MVN
        :param other:
        :return:
        """

        if isinstance(other, np.ndarray):
            return self.inv_transform(other, np.zeros(self.nb_dim))

        assert all([self.lmbda is not None, other.lmbda is not None]), "Precision not defined"

        if self.lmbda.ndim == 2:
            mu = self.lmbda.dot(self.mu) + other.lmbda.dot(other.mu)
        else:
            mu = np.einsum('aij, aj->ai', self.lmbda, self.mu) + np.einsum('aij, aj->ai', other.lmbda, other.mu)
        lmbda = self.lmbda + other.lmbda
        sigma = np.linalg.inv(lmbda)

        # prod.mu = prod.sigma.dot(prod.mu)
        mu = np.linalg.solve(lmbda, mu)

        prod = MVN(mu=mu, sigma=sigma)

        return prod

    def __rmul__(self, other):
        """
        Standart product of MVN
        :param other:
        :return:
        """
        if isinstance(other, float):
            mvn_ = type(self)(mu=other * self.mu, sigma=self.sigma)
            return mvn_

        return self.__mul__(other, self)

    def __mod__(self, other):
        """
        Product of transformed experts with elimination of pseudo-inverse

        :param other:
        :return:
        """

        prod = type(self)()
        prod.lmbda = self.lmbda + other.lmbda  # \SUM s_U^T * Q_s * S_U + R_s
        prod.sigma = np.linalg.inv(prod.lmbda)

        prod.mu = prod.sigma.dot(self.lmbdaT.dot(self.muT) + other.lmbdaT.dot(other.muT))
        # sigma_U * \SUM s_U^T * Q_s * (mu_U - s_xi*x0) + 0

        return prod

    def __floordiv__(self, other):
        """
        Product of transformed experts with elimination of pseudo-inverse

        :param other:
        :return:
        """

        prod = type(self)()
        prod.lmbda = self.lmbda + other.lmbda  # \SUM s_U^T * Q_s * S_U + R_s
        prod.lmbda_mu = self.lmbda_mu + other.lmbda_mu
        return prod

    def alpha_divergence(self, other, alpha=0.5):
        "https://mast.queensu.ca/~communications/Papers/gil-msc11.pdf"
        lmbda = np.linalg.inv(alpha * other.sigma + (1. - alpha) * self.sigma)

        r = 0.5 * (self.mu - other.mu).T.dot(lmbda).dot(self.mu - other.mu) - \
            1. / (2 * alpha * (alpha - 1.)) * (
                    -np.linalg.slogdet(lmbda)[1] \
                    - (1. - alpha) * np.linalg.slogdet(self.sigma)[1] \
                    - alpha * np.linalg.slogdet(other.sigma)[1])

        return r

    def sample(self, size=None):
        return np.random.multivariate_normal(self.mu, self.sigma, size=size)

    def sample(self, size=None):
        if self.mu.ndim == 1:
            return np.random.multivariate_normal(self.mu, self.sigma, size=size)
        else:
            eps = np.random.normal(size=self.mu.shape)
            return self.mu + np.einsum('aij,aj->ai ', np.linalg.cholesky(self.sigma), eps)

    def pdf(self, x):
        return mvn_pdf(x, self.mu[None], self.sigma_chol[None], self.lmbda[None])


import scipy.sparse as ss
import scipy.sparse.linalg as sl


class SparseMVN(MVN):
    @property
    def sigma(self):
        if self._sigma is None and not self._lmbda is None:
            self._sigma = sl.inv(self._lmbda)
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self.nb_dim = value.shape[0]
        self._lmbda = None
        self._sigma_chol = None
        self._sigma = value
        self._eta = None

    @property
    def lmbda(self):
        if self._lmbda is None and not self._sigma is None:
            self._lmbda = sl.inv(self._sigma)
        return self._lmbda

    @lmbda.setter
    def lmbda(self, value):
        self._sigma = None  # reset sigma
        self._sigma_chol = None
        self._lmbda = value
        self._eta = None

    # @profile
    def __mod__(self, other):
        """
        Product of transformed experts with elimination of pseudo-inverse

        :param other:
        :return:
        """

        prod = type(self)()
        prod.lmbda = self.lmbda + other.lmbda
        prod.sigma = sl.inv(prod.lmbda)
        prod.mu = prod.sigma.dot(
            self.lmbdaT.dot(self.muT) + other.lmbdaT.dot(other.muT))

        return prod
