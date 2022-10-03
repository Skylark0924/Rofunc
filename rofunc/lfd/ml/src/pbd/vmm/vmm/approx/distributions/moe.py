import tensorflow as tf
from tensorflow_probability import distributions as ds
from tensorflow_probability.python.internal import distribution_util as distribution_utils
from ...utils.tf_utils import *
from .gmm import GMMFull

class Gate(object):
	def __init__(self):
		pass

	@property
	def opt_params(self):
		raise NotImplementedError

	def conditional_mixture_distribution(self, x):
		"""
		x : [batch_shape, dim]
		return [batch_shape, nb_experts]
		"""
		raise NotImplementedError


class Experts(object):
	def __init__(self):
		pass

	@property
	def opt_params(self):
		raise NotImplementedError

	@property
	def nb_experts(self):
		raise NotImplementedError

	@property
	def nb_dim(self):
		raise NotImplementedError


	def conditional_components_distribution(self, x):
		"""
		x : [batch_shape, dim]
		return distribution([batch_shape, nb_experts, dim_out])
		"""
		raise NotImplementedError


class LinearMVNExperts(Experts):
	def __init__(self, A, b, cov_tril):
		self._A = A
		self._b = b
		self._cov_tril = cov_tril
		self._covs = matmatmul(self._cov_tril, self._cov_tril, transpose_b=True)

	@property
	def nb_dim(self):
		return self._b.shape[-1].value

	@property
	def nb_experts(self):
		return self._b.shape[0].value

	def conditional_components_distribution(self, x):
		ys = tf.einsum('aij,bj->abi', self._A, x) + self._b[:, None]

		return ds.MultivariateNormalTriL(
			tf.transpose(ys, perm=(1, 0, 2)),  # One for each component.
			self._cov_tril)

	@property
	def opt_params(self):
		return [self._A, self._b, self._cov_tril]

class GMMGate(Gate):
	def __init__(self, gmm):
		self._gmm = gmm

		Gate.__init__(self)

	@property
	def opt_params(self):
		return self._gmm.opt_params

	@property
	def gmm(self):
		return self._gmm

	def conditional_mixture_distribution(self, x):
		def logregexp(x, reg=1e-5, axis=0):
			return [x, reg * tf.ones_like(x)]

		return ds.Categorical(logits=tf.transpose(
			log_normalize(
				self._gmm.mixture_distribution.logits[:, None] +
				self._gmm.components_log_prob(x, name='marginal_prob_cond'),
						  axis=0)))


class MoE(object):
	def __init__(self, gate, experts, is_function=None):
		"""
		:type gate : Gate
		:type experts : Experts
		"""
		self._gate = gate
		self._experts = experts

		self._is_function = is_function

	@property
	def opt_params(self):
		return self._gate.opt_params + self._experts.opt_params

	@property
	def nb_experts(self):
		return self._experts.nb_experts

	def conditional_distribution(self, x):
		return ds.MixtureSameFamily(
			mixture_distribution=self._gate.conditional_mixture_distribution(x),
			components_distribution=self._experts.conditional_components_distribution(x)
		)


	def to_joint_marginal(self, y_sample, sess=None, nb_samples=20000):
		"""
		Create p(x, y) and p(x) given our MoE that is p(x | y)

		:param y_sample: a function that return samples of y as [batch_size, dim_y]
		:return:
		"""

		sess = sess if sess is not None else tf.get_default_session()

		y_ph = tf.placeholder(tf.float32, y_sample().shape)

		nb_samples_joint = 100

		state_idx = tf.argmax(self._gate.conditional_mixture_distribution(y_ph).logits,
							  axis=1)
		x = self.conditional_distribution(y_ph).sample(nb_samples_joint)

		_xs, _ys, _idxs = [], [], []

		for i in range(int(nb_samples/100)):
			_x, _y, _idx = sess.run([x, y_ph, state_idx],
									{y_ph: y_sample()})
			_xs += [_x]
			_ys += [_y]
			_idxs += [_idx]

		_xs = np.concatenate(_xs, axis=1)
		_ys = np.concatenate(_ys, axis=0)
		_idxs = np.concatenate(_idxs, axis=0)

		_priors_state = np.array([np.sum(_idxs == i) for i in range(self.nb_experts)],
								 dtype=np.float32)
		_priors_state /= np.sum(_priors_state)

		_ys_state = [_ys[_idxs == i] for i in range(self.nb_experts)]
		_xs_state = [np.concatenate(_xs[:, _idxs == i], axis=0) for i in
					 range(self.nb_experts)]


		log_priors = np.log(_priors_state + 1e-30)

		locs_x, locs_y, covs_x, covs_y = [], [], [], []

		for i in range(self.nb_experts):
			if np.sum(_idxs == i) > 2:
				locs_x += [np.mean(_xs_state[i], axis=0)]
				covs_x += [np.cov(_xs_state[i], rowvar=0)]

				locs_y += [np.mean(_ys_state[i], axis=0)]
				covs_y += [np.cov(_ys_state[i], rowvar=0)]
			else:
				locs_x += [np.zeros((x.shape[-1].value,))]
				locs_y += [np.zeros((y_ph.shape[-1].value,))]
				covs_x += [np.eye(x.shape[-1].value)]
				covs_y += [np.eye(y_ph.shape[-1].value)]

		locs_x, locs_y, covs_x, covs_y = np.array(locs_x), np.array(locs_y), np.array(covs_x), np.array(covs_y)


		locs_joint = np.concatenate([locs_x, locs_y], axis=1)

		covs_joint = np.zeros((self.nb_experts, locs_joint.shape[-1], locs_joint.shape[-1]))
		covs_joint[:, :locs_x.shape[-1], :locs_x.shape[-1]] = covs_x
		covs_joint[:, locs_x.shape[-1]:, locs_x.shape[-1]:] = covs_y

		marginal = GMMFull(
			log_priors=tf.convert_to_tensor(log_priors, dtype=tf.float32),
			locs=tf.convert_to_tensor(locs_x, dtype=tf.float32),
			tril_cov=tf.convert_to_tensor(np.linalg.cholesky(covs_x), dtype=tf.float32))

		joint = GMMFull(
			log_priors=tf.convert_to_tensor(log_priors, dtype=tf.float32),
			locs=tf.convert_to_tensor(locs_joint, dtype=tf.float32),
			tril_cov=tf.convert_to_tensor(np.linalg.cholesky(covs_joint), dtype=tf.float32))

		return joint, marginal



	#### FOLLOWING ARE WRONG BUT MAYBE COULD BE CORRECTED
	# def to_joint_distribution(self):
	# 	"""
	# 	Convert to joint distribution of [x, y]
	# 	:return:
	# 	"""
	# 	assert isinstance(self._gate, GMMGate), "Gate should be GMM"
	# 	assert isinstance(self._experts, LinearMVNExperts), "Experts should be linearMVN"
	#
	# 	loc_in = self._gate.gmm.locs
	# 	cov_in = self._gate.gmm.covs
	#
	# 	loc_out = self._experts._b + tf.einsum('aij,aj->ai', self._experts._A, loc_in)
	# 	cov_out = self._experts._covs
	#
	# 	# TODO check for cov_in_out
	# 	# cov_in_out = tf.matmul(self._experts._A, tf.linalg.inv(cov_in))
	# 	cov_in_out = tf.zeros_like(self._experts._A)
	#
	# 	loc = tf.concat([loc_in, loc_out], axis=1)
	#
	# 	cov = tf.concat([
	# 		tf.concat([cov_in, cov_in_out], axis=1),
	# 		tf.concat([tf.transpose(cov_in_out, perm=(0, 2, 1)), cov_out], axis=1)
	# 	], axis=2)
	#
	#
	# 	return GMMFull(
	# 		log_priors=self._gate.gmm.log_priors,
	# 		locs=loc,
	# 		covariance_matrix=cov
	# 	)
	#
	# def to_marginal_distribution(self):
	# 	"""
	# 	TODO should check if it is right or not
	# 	:return:
	# 	"""
	# 	assert isinstance(self._gate, GMMGate), "Gate should be GMM"
	# 	assert isinstance(self._experts, LinearMVNExperts), "Experts should be linearMVN"
	#
	# 	loc_in = self._gate.gmm.locs
	#
	# 	loc_out = self._experts._b + tf.einsum('aij,aj->ai', self._experts._A, loc_in)
	# 	cov_out = self._experts._covs
	#
	#
	# 	return GMMFull(
	# 		log_priors=self._gate.gmm.log_priors,
	# 		locs=loc_out,
	# 		covariance_matrix=cov_out
	# 	)


	def approximate_mode(self, x, return_distribution=False, return_idx=False):
		"""
		Get mean of most probable expert
		TODO implement better
		:param x:
		:return:
		"""

		cond = self.conditional_distribution(x)
		idx = tf.cast(tf.argmax(
				cond.mixture_distribution.logits,
				axis=1)[:, None],
					dtype=tf.int32)

		if return_distribution:

			parameters = {}

			for name, par in cond.components_distribution.parameters.iteritems():
				if isinstance(par, tf.Tensor) or isinstance(par, tf.Variable):
					if self._experts.__class__.__name__ == "LinearMVNExperts" and name == 'scale_tril':
						# Special case because scale_tril is not batch
						parameters[name] = tf.gather(
							par,
							idx[:, 0])

					else:
						parameters[name] = tf.batch_gather(
							par,
							idx)[:, 0]

				else:
					parameters[name] = par

			d = cond.components_distribution.__class__(**parameters)
		else:
			d = tf.batch_gather(
					cond.components_distribution.loc,
				idx)[:, 0]

		if return_idx:
			return d , idx
		else:
			return d

	def log_prob(self, y, x):
		return self.conditional_distribution(x).log_prob(y)

	def kl_qp(self, p_log_prob, x_in, samples_size=10, temp=1.):
		"""
		self is q: computes \int q \log (q/p)
		:param q_log_prob:
		:return:
		"""
		samples_conc = tf.reshape(
			tf.transpose(
				self._experts.conditional_components_distribution(x_in).sample(1)[0],
				perm=(1, 0, 2)),
			(x_in.shape[0].value * self.nb_experts, -1)
			# we perform only 1 sample for each conditional TODO increase
		)
		x_in_conc = tf.reshape(
			x_in[None] * tf.ones((self.nb_experts, 1, 1)),
			(x_in.shape[0].value * self.nb_experts, -1)
			# check where to add axis
		)

		log_qs = tf.reshape(self.log_prob(samples_conc, x_in_conc),
							(self.nb_experts, x_in.shape[0].value))
		log_ps = tf.reshape(temp * p_log_prob(samples_conc),
							(self.nb_experts, x_in.shape[0].value))

		return -tf.reduce_sum(tf.reduce_mean(
			(tf.exp(tf.transpose(self._gate.conditional_mixture_distribution(x_in).logits)))
			* (log_ps - log_qs), axis=1))


	def sample_is(self, x, n=1):
		mixture_distribution, mixture_components = self._gate.h(x), self._experts.f(x)

		y = mixture_components.sample(n)
		# npdt = y.dtype.as_numpy_dtype

		is_logits = self._is_function(mixture_distribution.logits)
		is_mixture_distribution = ds.Categorical(logits=is_logits)
		idx = is_mixture_distribution.sample(n)

		# TODO check if we should not renormalize mixture.logits - tf.stop_...

		weights = tf.batch_gather(mixture_distribution.logits - tf.stop_gradient(is_logits),
								  tf.transpose(idx))
		# TODO check axis
		weights = tf.batch_gather(
			log_normalize(mixture_distribution.logits - tf.stop_gradient(is_logits), axis=1),
								  tf.transpose(idx))

		if n == 1:
			return tf.batch_gather(y, idx[:, :, None])[0, :, 0], tf.transpose(weights)[0]
		else:
			return tf.batch_gather(y, idx[:, :, None])[:, :, 0], tf.transpose(weights)


