import numpy as np
from scipy.interpolate import interp1d
from scipy.special import gamma, gammaln

colvec = lambda x: np.array(x).reshape(-1, 1)
rowvec = lambda x: np.array(x).reshape(1, -1)

realmin = np.finfo(np.float64).tiny
realmax = np.finfo(np.float64).max


def limit_gains(gains, gain_limit):
    """

    :param gains:			[np.array]
    :param gain_limit 	[float]

    :return:
    """
    u, v = np.linalg.eig(gains)
    u[u > gain_limit] = gain_limit

    return v.dot(np.diag(u)).dot(np.linalg.inv(v))


def eigs(X):
    ''' Sorted eigenvalues and eigenvectors'''
    D, V = np.linalg.eig(X)
    sort_perm = D.argsort()[::-1]
    return D[sort_perm], V[:, sort_perm]


def mul(X):
    ''' Multiply an array of matrices'''
    x = X[0];
    for y in X[1:]:
        x = np.dot(x, y)
    return x


def spline(x, Y, xx, kind='cubic'):
    ''' Attempts to imitate the matlab version of spline'''
    from scipy.interpolate import interp1d
    if Y.ndim == 1:
        return interp1d(x, Y, kind=kind)[xx]
    F = [interp1d(x, Y[i, :]) for i in range(Y.shape[0])]
    return np.vstack([f(xx) for f in F])


def get_canonical_system(n_vars, dt):
    ''' Create a n_vars discrete canonical system with time step dt. '''
    # Continouse dynamical system:
    A = np.kron([[0, 1], [0, 0]], np.eye(n_vars))
    B = np.kron([[0], [1]], np.eye(n_vars))
    C = np.kron(np.eye(2), np.eye(n_vars))

    # Discretize:
    Ad = A * dt + np.eye(A.shape[0])
    Bd = B * dt

    return (Ad, Bd, C)


def get_dynamical_feature_matrix(n_varspos, n_derivs, n_data, n_samples, dt):
    '''Get the dynamical feature matrix that extracts n_derivs dynamical features from
    a n_varspos*n_data*n_samples vector of data points using dt as time discritization.

    Output: (PHI1,PHI,T1,T)
    o PHI1: Dynamical feature matrix for one sample
    o PHI : Dynamical feature matrix for n_samples
    '''

    T1 = n_data
    T = n_data * n_samples

    # Create op Matrix for one dimension:
    op1D = np.zeros((n_derivs, n_derivs))
    op1D[0, n_derivs - 1] = 1

    for i in range(1, n_derivs):
        op1D[i, :] = (op1D[i - 1,] - np.roll(op1D[i - 1,], -1)) / dt

    # Extend to other dimensions
    # put the operator in a big matrix that we
    # use for the roll operation
    op = np.zeros((T1 * n_derivs, T1))
    i1 = (n_derivs - 1) * n_derivs
    i2 = n_derivs * n_derivs
    op[i1:i2, 0:n_derivs] = op1D;

    PHI0 = np.zeros((T1 * n_derivs, T1))

    # Create Phi
    for t in range(0, T1 - n_derivs + 1):
        tmp = np.roll(op, t * n_derivs, axis=0)  # Shift in the first dimension
        tmp = np.roll(tmp, t, axis=1)  # Shift in the second dimension
        PHI0 = PHI0 + tmp  # Add to PHI

    # Handle borders:
    for i in range(1, n_derivs):
        op[n_derivs * n_derivs - i,] = 0
        op[:, i - 1] = 0
        tmp = np.roll(op, -i * n_derivs, axis=0)
        tmp = np.roll(tmp, -i, axis=1)
        PHI0 = PHI0 + tmp
    # Construct Phi matrices:
    PHI1 = np.kron(PHI0, np.eye(n_varspos))
    PHI = np.kron(np.eye(n_samples), PHI1)

    return PHI1, PHI


def condition_gaussian(Mu, Sigma, sample, input, output):
    slii = np.ix_(input, input)
    sloi = np.ix_(output, input)
    sloo = np.ix_(output, output)
    slio = np.ix_(input, output)

    InvSigmaInIn = np.linalg.inv(Sigma[slii])
    InvSigmaOutIn = np.dot(Sigma[sloi], InvSigmaInIn)

    MuOut = Mu[output] + np.dot(InvSigmaOutIn,
                                (sample - Mu[input]).T)

    SigmaOut = Sigma[sloo] - \
               np.dot(InvSigmaOutIn, \
                      Sigma[slio])
    return MuOut, SigmaOut


def get_state_prediction_matrix(A, B, Np, **kwargs):
    ''' Returns matrix to be used for batch prediction of the state of the discrete system
    x_k+1 = A*x_k + B*u_k
    '''
    # Check if number of control predictions was specified,
    # if not take the same as specified Np
    Nc = kwargs.get('Nc', Np)

    # Get dimensions:
    (nA, mA) = A.shape
    (_, mB) = B.shape

    # Construct Sx:
    Sx = np.zeros((nA * Np, mA))
    c1 = np.zeros((nA * Np, mB))
    Sx[0:nA, ] = A
    c1[0:nA, ] = B

    for kk in range(1, Np):
        ind1 = slice((kk - 1) * nA, kk * nA, 1)
        ind2 = slice(kk * nA, (kk + 1) * nA, 1)
        Sx[ind2, :] = Sx[ind1, :].dot(A)
        c1[ind2, :] = Sx[ind1, :].dot(B)

    Su = np.zeros((Np * nA, mB * Nc))
    for kk in range(0, Nc):
        rInd1 = kk * nA
        rInd2 = (Np - kk) * nA
        cInd = slice(kk * mB, (kk + 1) * mB, 1)

        Su[rInd1::, cInd] = c1[0:rInd2, :]

    return (Su, Sx)


def multi_variate_normal_old(x, mean, covar):
    '''Multi-variate normal distribution

    x: [n_data x n_vars] matrix of data_points for which to evaluate
    mean: [n_vars] vector representing the mean of the distribution
    covar: [n_vars x n_vars] matrix representing the covariance of the distribution

    '''

    # Check dimensions of covariance matrix:
    if type(covar) is np.ndarray:
        n_vars = covar.shape[0]
    else:
        n_vars = 1

    # Check dimensions of data:
    if x.ndim == 1 and n_vars == len(x):
        n_data = 1
    else:
        n_data = x.shape[0]

    diff = (x - mean).T

    # Distinguish between multi and single variate distribution:
    if n_vars > 1:
        lambdadiff = np.linalg.inv(covar).dot(diff)
        scale = np.sqrt(
            np.power((2 * np.pi), n_vars) * (abs(np.linalg.det(covar)) + 1e-200))
        p = np.sum(diff * lambdadiff, 0)
    else:
        lambdadiff = diff / covar
        scale = np.sqrt(np.power((2 * np.pi), n_vars) * covar + 1e-200)
        p = diff * lambdadiff

    return np.exp(-0.5 * p) / scale


def prod_gaussian(mu_1, sigma_1, mu_2, sigma_2):
    prec_1 = np.linalg.inv(sigma_1)
    prec_2 = np.linalg.inv(sigma_2)
    # Compute covariance of p	roduct:

    Sigma = np.linalg.inv(prec_1 + prec_2)

    # Compute mean of product:
    Mu = Sigma.dot(prec_1.dot(mu_1) + prec_2.dot(mu_2))

    return Mu, Sigma


def mvn_pdf(x, mu, sigma_chol, lmbda, sigma=None, reg=None):
    """

    :param x: 			np.array([nb_dim x nb_samples])
        samples
    :param mu: 			np.array([nb_states x nb_dim])
        mean vector
    :param sigma_chol: 	np.array([nb_states x nb_dim x nb_dim])
        cholesky decomposition of covariance matrices
    :param lmbda: 		np.array([nb_states x nb_dim x nb_dim])
        precision matrices
    :return: 			np.array([nb_states x nb_samples])
        log mvn
    """
    N = mu.shape[0]
    D = mu.shape[1]

    if len(x.shape) > 1:  # TODO implement mvn for multiple xs
        raise NotImplementedError

    dx = mu - x

    if reg is not None:
        if isinstance(reg, list):
            reg = np.power(np.diag(reg), 2)
        else:
            reg = np.power(reg * np.eye(D), 2)

        lmbda_ = np.linalg.inv(np.linalg.inv(lmbda) + reg)
        sigma_chol_ = sigma_chol + reg
    else:
        lmbda_ = lmbda
        sigma_chol_ = sigma_chol

    return -0.5 * np.einsum('...j,...j', dx, np.einsum('...jk,...j->...k', lmbda_, dx)) \
           - D / 2 * np.log(2 * np.pi) - np.sum(np.log(sigma_chol_.diagonal(axis1=1, axis2=2)), axis=1)


# return np.asarray([-0.5 * dx[i].T.dot(lmbda[i]).dot(dx[i]) \
# 				   - D / 2 * np.log(2 * np.pi) - np.sum(
# 	np.log(sigma_chol[i].diagonal(axis1=0, axis2=1)), axis=0)
# 				   for i in range(N)])

def multi_variate_t(x, nu, mu, sigma=None, log=True, gmm=False, lmbda=None):
    """
    Multivariatve T-distribution PDF
    https://en.wikipedia.org/wiki/Multivariate_t-distribution

    :param x:		np.array([nb_samples, nb_dim])
    :param mu: 		np.array([nb_dim])
    :param sigma: 	np.array([nb_dim, nb_dim])
    :param log: 	bool
    :return:
    """
    from scipy.special import gamma
    if not gmm:
        if type(sigma) is float:
            sigma = np.array(sigma, ndmin=2)
        if type(mu) is float:
            mu = np.array(mu, ndmin=1)
        if sigma is not None:
            sigma = sigma[None, None] if sigma.shape == () else sigma

        mu = mu[None] if mu.shape == () else mu
        x = x[:, None] if x.ndim == 1 else x

        p = mu.shape[0]

        dx = mu - x
        lmbda_ = np.linalg.inv(sigma) if lmbda is None else lmbda

        dist = np.einsum('...j,...j', dx, np.einsum('...jk,...j->...k', lmbda_, dx))
        # (nb_timestep, )

        if not log:
            lik = gamma((nu + p) / 2) * np.linalg.det(lmbda_) ** 0.5 / \
                  (gamma(nu / 2) * nu ** (p / 2) * np.pi ** (p / 2)) * \
                  (1 + 1 / nu * dist) ** (-(nu + p) / 2)
            return lik
        else:
            log_lik = gammaln((nu + p) / 2) + 0.5 * np.linalg.slogdet(lmbda_)[1] - \
                      gammaln(nu / 2) - p / 2. * (np.log(nu) + np.log(np.pi)) + \
                      ((-(nu + p) / 2) * np.log(1 + dist / nu))

            return log_lik
    else:
        raise NotImplementedError


def multi_variate_normal(x, mu, sigma=None, log=True, gmm=False, lmbda=None):
    """
    Multivariatve normal distribution PDF

    :param x:		np.array([nb_samples, nb_dim])
    :param mu: 		np.array([nb_dim])
    :param sigma: 	np.array([nb_dim, nb_dim])
    :param log: 	bool
    :return:
    """
    if not gmm:
        if type(sigma) is float:
            sigma = np.array(sigma, ndmin=2)
        if type(mu) is float:
            mu = np.array(mu, ndmin=1)
        if sigma is not None:
            sigma = sigma[None, None] if sigma.shape == () else sigma

        mu = mu[None] if mu.shape == () else mu
        x = x[:, None] if x.ndim == 1 else x

        dx = mu - x
        lmbda_ = np.linalg.inv(sigma) if lmbda is None else lmbda

        log_lik = -0.5 * np.einsum('...j,...j', dx, np.einsum('...jk,...j->...k', lmbda_, dx))

        if sigma is not None:
            log_lik -= 0.5 * (x.shape[1] * np.log(2 * np.pi) + np.linalg.slogdet(sigma)[1])
        else:
            log_lik -= 0.5 * (x.shape[1] * np.log(2 * np.pi) - np.linalg.slogdet(lmbda)[1])

        return log_lik if log else np.exp(log_lik)
    else:
        raise NotImplementedError

# # Check dimension of data
# if x.ndim == 1:
# 	n_vars = 1
# 	n_data = len(x)
# else:
# 	n_vars, n_data = x.shape
#
# if type(covar) == float:
# 	covar = np.eye(n_vars) * covar
#
# diff = x - mean[:, None]
#
# # Distinguish between multi and single variate distribution:
# if n_vars > 1:
# 	lambdadiff = np.linalg.inv(covar).dot(diff)
# 	scale = np.sqrt(
# 		np.power((2 * np.pi), n_vars) * (abs(np.linalg.det(covar)) + 1e-200))
# 	p = np.sum(diff * lambdadiff, 0)
# else:
# 	lambdadiff = diff / covar
# 	scale = np.sqrt(np.power((2 * np.pi), n_vars) * covar + 1e-200)
# 	p = diff * lambdadiff
#
# if not log:
# 	return np.exp(-0.5 * p) / scale
# else:
# 	return -0.5 * p - np.log(scale)
