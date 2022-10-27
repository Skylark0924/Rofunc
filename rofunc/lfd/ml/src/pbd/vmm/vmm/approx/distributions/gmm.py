import tensorflow as tf
from tensorflow_probability import distributions as ds
from ...utils.tf_utils import matmatmul
from ...utils import plot_utils

def log_normalize(x):
	return x - tf.reduce_logsumexp(x)

def param_from_pbd(model, dtype=tf.float32, cst=False, tril=False, std_diag=False,
				   student_t=False):
	conv = tf.Variable if not cst else tf.constant
	import numpy as np

	log_priors = conv(np.log(model.priors), dtype=dtype)
	locs = conv(model.mu, dtype=dtype)

	if tril:
		covs = conv(np.linalg.cholesky(model.sigma), dtype=dtype)
	elif std_diag:
		covs = conv(np.diag(model.sigma)**0.5, dtype=dtype)
	else:
		covs = conv(model.sigma, dtype=dtype)

	if not student_t:
		return log_priors, locs, covs
	else:
		return log_priors, conv(model.nu, dtype=dtype), locs, covs


class GMMDiag(ds.MixtureSameFamily):
	def __init__(self,
				 log_priors,
				 locs,
				 log_std_diags,
				 name='gmm_diag'):
		"""

		:param locs: 	Means of each components
		:type locs: 				[k, ndim]
		:param log_std_diags: log standard deviation of each component
			(log of diagonal scale matrix)
			diag(exp(log_std_diags)) diag(exp(log_std_diags))^T = cov
		:type log_std_diags: 		[k, ndim]
		"""
		self.k = locs.shape[0].value
		self.d = locs.shape[-1].value
		self.log_unnormalized_priors = log_priors
		self.locs = locs
		self.log_std_diags = log_std_diags

		self.log_priors = log_normalize(self.log_unnormalized_priors)
		self.priors = tf.exp(log_normalize(self.log_priors))

		self.covs = tf.matrix_diag(tf.exp(log_std_diags)**2)

		self._lin_op = tf.linalg.LinearOperatorDiag(tf.exp(log_std_diags))

		self.precs = tf.matrix_diag(tf.exp(-2 * log_std_diags))

		ds.MixtureSameFamily.__init__(
			self,
			mixture_distribution=ds.Categorical(logits=self.log_priors),
			components_distribution=ds.MultivariateNormalDiag(
				loc=locs, scale_diag=tf.exp(log_std_diags)
			)
		)


	@property
	def opt_params(self):
		return [self.log_unnormalized_priors, self.log_std_diags, self.locs]


	def components_log_prob(self, x, name='log_prob_comps'):
		"""
		Get log prob of every components

		x : [batch_shape, d]
		return [nb_components, batch_shape]
		"""
		return tf.transpose(self.components_distribution.log_prob(x[:, None], name=name))


	def component_sample(self, k, sample_shape=(), name='component_sample'):
		if isinstance(k, int):

			raise NotImplementedError("Reimplement this as it becomes a MixtureSameFamily")
			return self.components[k].sample(sample_shape=sample_shape, name=name)
		else:
			return ds.MultivariateNormalDiag(
				loc=self.locs[k], scale_diag=tf.exp(self.log_std_diags[k])).sample(sample_shape=sample_shape, name=name)

	def all_components_sample(self, sample_shape=(), name='components_sample'):
		return self.components_distribution.sample(sample_shape, name=name)

	def plot(self, sess=None, feed_dict={}, *args, **kwargs):
		if sess is None:
			sess = tf.get_default_session()

		alpha_priors = kwargs.pop('alpha_priors', False)
		if not alpha_priors:
			mu, sigma = sess.run([self.locs, self.covs], feed_dict=feed_dict)
		else:
			mu, sigma, alpha = sess.run([self.locs, self.covs, self.priors], feed_dict=feed_dict)
			kwargs['alpha'] = (alpha + 0.02) /max(alpha + 0.02)


		plot_utils.plot_gmm(mu, sigma, **kwargs)

	def product_int(self, get_dist=False):
		"""
		Compute log \int_x N_1(x) N_2(x) dx for any two Gaussians

		:param get_dist: if True return only positive number that can be interpreted as distances
		"""
		m = ds.MultivariateNormalFullCovariance(
			self.locs[None], self.covs[None] + self.covs[:, None]).log_prob(
			self.locs[:, None])

		if not get_dist:
			return m
		else:
			return -m - tf.reduce_min(-m)

	def kl_divergence_components(self):
		"""
		Compute kl divergence between any two components
		:return: [k, k]
		"""
		log_det = tf.linalg.slogdet(self.covs)[1]

		return tf.reduce_sum(
			tf.einsum('abi,aij->abj', self.locs[None] - self.locs[:, None],
					  tf.linalg.inv(self.covs))
			* (self.locs[None] - self.locs[:, None]),
			axis=2
		) + \
			   (log_det[:, None] - log_det[None]) + \
			   tf.einsum('aij,bji->ab', tf.linalg.inv(self.covs), self.covs) - \
			   tf.cast(self.event_shape[0], tf.float32)

	def kl_qp(self, p_log_prob, samples_size=10, temp=1.):
		"""
		self is q: computes \int q \log (q/p)
		:param q_log_prob:
		:return:
		"""

		samples_conc = tf.reshape(
			tf.transpose(self.all_components_sample(samples_size), perm=(1, 0, 2))
			, (samples_size * self.k, -1))  # [k * nsamples, ndim]

		log_qs = tf.reshape(self.log_prob(samples_conc), (self.k, samples_size))
		log_ps = tf.reshape(temp * p_log_prob(samples_conc),
							(self.k, samples_size))

		component_elbos = tf.reduce_mean(log_ps - log_qs, axis=1)

		return -tf.reduce_sum(component_elbos * self.priors)

	def approximxate_mode(self, only_priors=False):
		if only_priors:
			idx = tf.argmax(self.log_priors)
		else:
			idx = tf.argmax(
				self.components_distribution.log_prob(self.locs) + self.log_priors
			)
		return tf.gather(self.locs, idx)


	def entropy(self, samples_size=20):
		samples_conc = tf.reshape(
			tf.transpose(self.all_components_sample(samples_size), perm=(1, 0, 2))
			, (samples_size * self.k, -1))  # [k * nsamples, ndim]

		log_qs = tf.reshape(self.log_prob(samples_conc), (self.k, samples_size))

		return tf.reduce_sum(tf.reduce_mean(log_qs, axis=1) * self.priors)

	def kl_pq(self, p, samples_size=10, temp=1.):
		"""
		self is q: computes \int p \log (p/q)
		:param p: Distribution should have p.sample(N) and p.log_prob()
		:return:
		"""
		#
		# if isinstance(p, ds.Mixture):
		# 	raise NotImplementedError

		samples = p.sample(samples_size)

		log_ps = p.log_prob(samples)
		log_qs = temp * self.log_prob(samples)

		return -tf.reduce_sum(log_qs-log_ps)

class MixtureMeanField(GMMDiag):
	def __init__(self, qs, name='MixtureMeanField'):
		self._qs = qs

		self._name = name
		self._dtype = qs[0].dtype

		self.k = qs[0].mixture_distribution._logits.shape[0].value

		self._qs_slices = []
		self._qs_d = []
		e = 0
		for q in qs:
			size = q.event_shape[0].value if len(q.event_shape) else 1
			self._qs_slices += [slice(e, e + size)]
			self._qs_d += [size]
			e += size

		self._use_static_graph = False
		self._assertions = []

		# self._static_event_shape = qs[0]._static_event_shape
		# self._static_batch_shape = qs[0]._static_batch_shape
		self._assertions = []
		self._runtime_assertions = []
		graph_parents = []
		for q in qs:
			graph_parents += q._graph_parents  # pylint: disable=protected-access

		self._graph_parents = graph_parents

	@property
	def priors(self):
		return tf.exp(self._qs[0].mixture_distribution._logits)

	@property
	def log_priors(self):
		return self._qs[0].mixture_distribution._logits

	def components_log_prob(self, x, name='log_prob_comps'):
		"""
		Get log prob of every components

		x : [batch_shape, d]
		return [nb_components, batch_shape]
		"""

		qs_components_log_prob = []

		for i, q in enumerate(self._qs):
			if self._qs_d[i] > 1: # vector distribution
				_l = q.components_distribution.log_prob(x[:, None, self._qs_slices[i]], name=name)
			else: # scalar distribution
				_l = q.components_distribution.log_prob(x[:, None, self._qs_slices[i].start], name=name)

			qs_components_log_prob += [_l]

		return tf.transpose(
			tf.reduce_sum(
				qs_components_log_prob,
				axis=0)
		)

	def log_prob(self, value, name="log_prob"):
		return tf.reduce_logsumexp(
			self.components_log_prob(value, name=name) + self.log_priors[:, None], axis=0)

	def sample(self, sample_shape=(), seed=None, name="sample"):
		samples = []

		for q in self._qs:
			sample = q.sample(sample_shape, seed=seed, name=name)
			if sample.shape.ndims == 1:
				sample = sample[:, None]
			samples += [sample]

		return tf.concat(
			samples, axis=-1)

	def all_components_sample(self, sample_shape=(), name='components_sample'):
		samples = []

		for q in self._qs:
			if hasattr(q, 'all_components_sample'):
				sample = q.all_components_sample(sample_shape, name=name)
			elif hasattr(q, 'components_distribution'):
				sample = q.components_distribution.sample(sample_shape, name=name)
			else:
				raise NotImplementedError

			if sample.shape.ndims == 2:
				sample = sample[:, :, None]
			samples += [sample]

		return tf.concat(
			samples, axis=-1)


class GMMFull(GMMDiag):
	def __init__(self,
				 log_priors,
				 locs,
				 tril_cov=None,
				 covariance_matrix=None,
				 name='gmm_full'):
		"""

		:param locs: 	Means of each components
		:type locs: 				[k, ndim]
		:param log_std_diags: log standard deviation of each component
			(log of diagonal scale matrix)
			diag(exp(log_std_diags)) diag(exp(log_std_diags))^T = cov
		:type log_std_diags: 		[k, ndim]
		"""
		self.k = locs.shape[0] if isinstance(locs.shape[0], int) else locs.shape[0].value
		self.d = locs.shape[-1]  if isinstance(locs.shape[-1], int) else locs.shape[-1].value

		self.log_unnormalized_priors = log_priors
		self.locs = locs

		self.log_priors = log_normalize(self.log_unnormalized_priors)
		self.priors = tf.exp(log_normalize(self.log_priors))

		if tril_cov is not None:
			self.tril_cov = tril_cov
		elif covariance_matrix is not None:
			self.covs = covariance_matrix
			self.tril_cov = tf.linalg.cholesky(self.covs)
		else:
			raise ValueError("Either tril_cov or covariance matrix should be specified")

		self.tril_precs = tf.linalg.LinearOperatorLowerTriangular(
			self.tril_cov).solve(tf.eye(self.d))


		self.precs = matmatmul(self.tril_precs, self.tril_precs, transpose_a=True)

		if tril_cov is not None:
			_tril_cov = self.tril_cov[None] if self.tril_cov.shape.ndims == 2 else self.tril_cov


			ds.MixtureSameFamily.__init__(
				self,
				mixture_distribution=ds.Categorical(logits=self.log_priors),
				components_distribution=ds.MultivariateNormalTriL(
					loc=self.locs, scale_tril=_tril_cov
				)
			)

			self.covs = self.components_distribution.covariance()
		else:
			ds.MixtureSameFamily.__init__(
				self,
				mixture_distribution=ds.Categorical(logits=self.log_priors),
				components_distribution=ds.MultivariateNormalFullCovariance(
					loc=self.locs, covariance_matrix=self.covs
				)
			)


	@property
	def opt_params(self):
		return [self.log_unnormalized_priors, self.tril_cov, self.locs]


	def component_sample(self, k, sample_shape=(), name='component_sample'):
		if isinstance(k, int):
			raise NotImplementedError("Reimplement this as it becomes a MixtureSameFamily")
			return self.components[k].sample(sample_shape=sample_shape, name=name)
		else:
			return ds.MultivariateNormalTriL(
				loc=self.locs[k], scale_tril=tf.exp(self.tril_cov[k])).sample(sample_shape=sample_shape, name=name)
