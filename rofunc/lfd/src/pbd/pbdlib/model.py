from .functions import *
from .utils.gaussian_utils import gaussian_moment_matching
from .plot import plot_gmm



class Model(object):
	"""
	Basis class for Gaussian mixture model (GMM), Hidden Markov Model (HMM), Hidden semi-Markov
	Model (HSMM), Product of Gaussian mixture model (PoGMM)
	"""

	def __init__(self, nb_states, nb_dim=2):
		self._priors = None
		self.nb_dim = nb_dim
		self.nb_states = nb_states

		self._mu = None
		self._sigma = None  # covariance matrix
		self._sigma_chol = None  # covariance matrix, cholesky decomposition
		self._lmbda = None  # Precision matrix
		self._eta = None

		self._reg = None
		self.nb_dim = nb_dim

		self._has_finish_state = False
		self._has_init_state = False

		self._log_normalization = None

	@property
	def has_finish_state(self):
		return self._has_finish_state

	@property
	def has_init_state(self):
		return self._has_init_state

	def regularize(self, reg):
		"""

		:param reg: is as
			float: isotopric std deviation
			np.array() rank 1: std deviation diagonal
			np.array() rank 2: covariance matrix

		:return:
		"""
		new_model = Model(nb_states=self.nb_states, nb_dim=self.nb_dim)
		new_model.mu = self.mu

		if isinstance(reg, float):
			new_model.sigma = self.sigma + np.eye(self.sigma.shape[-1])[None] * reg ** 2
		elif isinstance(reg, np.ndarray) and reg.ndim == 1:
			new_model.sigma = self.sigma + np.diag(reg**2)[None]
		else:
			new_model.sigma = self.sigma + reg[None]

		return new_model

	@property
	def reg(self):
		"""
		Regularization term. Diagonal matrix added to covariance to reduce overfitting.
		Is only used during learning procedure.

		:return: [np.array([nb_dim, nb_dim])]
		"""
		if self._reg is None:
			self._reg = 1e-10 * np.eye(self.nb_dim)
		return self._reg

	@reg.setter
	def reg(self, value):
		if value is None:
			self._reg = 1e-10 * np.eye(self.nb_dim)
		elif isinstance(value, list):
			self._reg = np.diag(value) ** 2
		elif isinstance(value, np.ndarray):
			self._reg = value
		elif isinstance(value, float):
			print(value)
			self._reg = value ** 2 * np.eye(self.nb_dim)
		else:
			raise ValueError('Regularization should be of type float, ndarray or list')

	@property
	def priors(self):
		"""
		Priors distributions for GMM

		:return:	np.array([nb_states])
		"""
		return self._priors

	@priors.setter
	def priors(self, value):
		self._priors = value

	@property
	def log_priors(self):
		if self._log_priors is None:
			self._log_priors = np.log(self._priors + 1e-40)

		return self._log_priors

	@log_priors.setter
	def log_priors(self, value):
		self._log_priors = value

	@property
	def mu(self):
		"""
		Mean of MVNs distributions

		:return: [np.array([nb_states, nb_dim])]
		"""
		return self._mu

	@mu.setter
	def mu(self, value):
		self.nb_dim = value.shape[-1]
		self.nb_states = value.shape[0]
		self._mu = value

	@property
	def eta(self):
		if self._eta is None:
			self._eta = np.einsum('aij,aj->ai', self.lmbda, self.mu)

		return self._eta

	@property
	def sigma_chol(self):
		"""
		Cholesky decomposition of covariance matrices of MVNs distributions

		:return: [np.array([nb_states, nb_dim, nb_dim])]
		"""
		if self.sigma is None:
			return None
		else:
			if self._sigma_chol is None:
				self._sigma_chol = np.linalg.cholesky(self.sigma)
			return self._sigma_chol

	@property
	def sigma(self):
		"""
		Covariance matrices of MVNs distributions

		:return: [np.array([nb_states, nb_dim, nb_dim])]
		"""
		if self._sigma is None and not self._lmbda is None:
			self._sigma = np.linalg.inv(self._lmbda)
		return self._sigma

	@sigma.setter
	def sigma(self, value):
		self._eta = None
		self._lmbda = None
		self._sigma_chol = None
		self._sigma = value
		self._log_normalization = None

	@property
	def lmbda(self):
		"""
		Precision matrices (inverse of covariance) of MVNs distributions

		:return: [np.array([nb_states, nb_dim, nb_dim])]
		"""
		if self._lmbda is None and not self._sigma is None:
			self._lmbda = np.linalg.inv(self._sigma)
		return self._lmbda

	@lmbda.setter
	def lmbda(self, value):
		self._eta = None
		self._sigma = None  # reset sigma
		self._sigma_chol = None
		self._lmbda = value
		self._log_normalization = None

	def get_dep_mask(self, deps):
		mask = np.eye(self.nb_dim)

		for dep in deps:
			if isinstance(dep, slice):
				mask[dep, dep] = 1.
			elif isinstance(dep, list):
				dGrid = np.ix_(dep, dep)
				mask[dGrid] = 1.

		return mask

	def dep_mask(self, deps):
		"""
		Remove covariances between elements in the covariance matrix.

		:param deps: 		[list of slices]
		List of slices of block-diagonal parts to keep. i.e. [slice(0,2), slice(2,4)]
		:return:
		"""

		mask = self.get_dep_mask(deps)

		self.sigma *= mask

		# reset parameters
		self._lmbda = None
		self._sigma_chol = None

	def keeponlydims(self, sl):
		"""
		Remove some dimensions of the model
		:param sl: 	[slice]
			Slice of the dimensions to keep
		:return:
		"""
		self.mu = self.mu[:, sl]
		self.sigma = self.sigma[:, sl, sl]

	def init_zeros(self):
		"""
		Init all parameters
		:return:
		"""
		self._priors = np.ones(self.nb_states)/self.nb_states
		self._mu = np.array([np.zeros(self.nb_dim) for i in range(self.nb_states)])
		self._sigma = np.array([np.eye(self.nb_dim) for i in range(self.nb_states)])

	def plot(self, *args, **kwargs):
		"""
		Plot GMM, circle is 1 std

		:param args:
		:param kwargs:
		:return:
		"""
		plot_gmm(self.mu, self.sigma, *args, swap=True, **kwargs)

	def sample(self, size=1):
		"""
		Generate random samples from GMM
		:param size: 	[int]
		:return:
		"""
		zs = np.array([np.random.multinomial(1, self.priors) for _ in range(size)]).T

		xs = [z[:, None] * np.random.multivariate_normal(m, s, size=size) for z, m, s in zip(zs, self.mu, self.sigma)]

		return np.sum(xs, axis=0)

	def get_linear_conditional(self, dim_in, dim_out):
		mu_in, sigma_in = self.get_marginal(dim_in)
		mu_out, sigma_out = self.get_marginal(dim_out)  # get marginal distribution of x_out

		# get conditional distribution of x_out given x_in for each states p(x_out|x_in, k)
		_, sigma_in_out = self.get_marginal(dim_in, dim_out)

		inv_sigma_in_in = np.linalg.inv(
			sigma_in)
		inv_sigma_out_in = np.einsum('aji,ajk->aik', sigma_in_out, inv_sigma_in_in)

		As = inv_sigma_out_in
		bs = mu_out - np.matmul(inv_sigma_out_in, mu_in[:, :, None])[:, :, 0]

		sigma_est = (sigma_out - np.matmul(inv_sigma_out_in, sigma_in_out))

		return As, bs, sigma_est

	def condition(self, data_in, dim_in, dim_out, h=None, return_gmm=False):
		"""
		:param data_in:		[np.array([nb_timestep, nb_dim])
		:param dim_in:
		:param dim_out:
		:param h:
		:return:
		"""
		sample_size = data_in.shape[0]

		# compute responsabilities
		mu_in, sigma_in = self.get_marginal(dim_in)

		if h is None:
			h = np.zeros((self.nb_states, sample_size))
			for i in range(self.nb_states):
				h[i, :] = multi_variate_normal(data_in,
											   mu_in[i],
											   sigma_in[i])

			h += np.log(self.priors)[:, None]
			h = np.exp(h).T
			h /= (np.sum(h, axis=1, keepdims=True) + realmin)
			h = h.T

		# self._h = h
		mu_out, sigma_out = self.get_marginal(dim_out)
		mu_est, sigma_est = ([], [])

		inv_sigma_in_in, inv_sigma_out_in = ([], [])

		_, sigma_in_out = self.get_marginal(dim_in, dim_out)

		for i in range(self.nb_states):
			inv_sigma_in_in += [np.linalg.inv(sigma_in[i])]
			inv_sigma_out_in += [sigma_in_out[i].T.dot(inv_sigma_in_in[-1])]

			mu_est += [mu_out[i] + np.einsum('ij,aj->ai', inv_sigma_out_in[-1], data_in - mu_in[i])]

			sigma_est += [sigma_out[i] - inv_sigma_out_in[-1].dot(sigma_in_out[i])]

		mu_est, sigma_est = (np.asarray(mu_est), np.asarray(sigma_est))

		if return_gmm:
			if sample_size == 1:
				from .gmm import GMM
				return GMM(priors=h[:, 0], mu=mu_est[:, 0], sigma=sigma_est,
						   nb_dim=mu_est.shape[-1], nb_states=mu_est.shape[0])
			return h, mu_est, sigma_est
		# return np.mean(mu_est, axis=0)
		else:

			return gaussian_moment_matching(mu_est, sigma_est, h.T)
			# return np.sum(h[:, :, None] * mu_est, axis=0), np.sum(
			# 	h[:, :, None, None] * sigma_est[:, None], axis=0)

	def get_marginal(self, dim, dim_out=None, get_eta=False, get_lmbda=False):
		"""
		Get marginal model or covariance between blocks of variables

		:param dim: 		[slice] or [list of index]
		:param dim_out: 	[slice] or [list of index]
		:return:
		"""
		mu, sigma, eta = (self.mu, self.sigma, self.eta)
		if get_lmbda:
			mu, sigma, eta = (self.mu, self.lmbda, self.eta)

		if isinstance(dim, list):
			if dim_out is not None:
				dGrid = np.ix_(range(self.nb_states), dim, dim_out)
			else:
				dGrid = np.ix_(range(self.nb_states), dim, dim)

			mu, sigma = (mu[:, dim], sigma[dGrid])

		elif isinstance(dim, slice):
			if dim_out is not None:
				mu, sigma = (mu[:, dim], sigma[:, dim, dim_out])
			else:
				mu, sigma = (mu[:, dim], sigma[:, dim, dim])

		if get_eta:
			return mu, sigma, eta[:, dim]
		else:
			return mu, sigma


