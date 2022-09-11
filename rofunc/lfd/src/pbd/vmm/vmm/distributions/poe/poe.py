import tensorflow as tf
from tensorflow_probability import distributions as ds

class PoE(ds.Distribution):
	def __init__(self, shape, experts, transfs, name='PoE', cost=None):
		"""

		:param shape:
		:param experts:
		:param transfs: 	a list of tensorflow function that goes from product space to expert space
			 or a function f(x: tensor, i: index of transform) -> f_i(x)

		:param cost: additional cost [batch_size, n_dim] -> [batch_size, ]
			a function f(x) ->
		"""

		self._product_shape = shape
		self._experts = experts
		self._transfs = transfs
		self._transfs_f = None
		self._laplace = None
		self._samples_approx = None

		self._cost = cost


		self.stepsize = tf.Variable(0.01)
		self._name = name


	@property
	def product_shape(self):
		return self._product_shape

	@property
	def experts(self):
		return self._experts

	@property
	def transfs(self):
		return self._transfs

	@property
	def transfs_f(self):
		if self._transfs_f is None:
			self._transfs_f = []

			if isinstance(self.transfs, list):
				for tr in self.transfs:
					self._transfs_f += [tr]
			else:
				self._transfs_f = self._transfs

		return self._transfs_f

	def _log_prob_sg(self, x):
		"""
		A function whose gradient is approximating the gradient of the normalized pdf.
		This function can be passed to an optimizer that uses gradients
		(and not the function itself, as linesearch)
		:param x:
		:return:
		"""
		return self._log_unnormalized_prob(x) - self._log_normalization_sg(x)

	def log_prob_sg(self, x):
		return self._log_prob_sg(x)

	def _experts_probs(self, x):
		probs = []

		for i, exp in enumerate(self.experts):
			if isinstance(self.transfs_f, list):
				if hasattr(exp, '_log_unnormalized_prob'):
					probs += [exp._log_unnormalized_prob(self.transfs_f[i](x))]
				else:
					probs += [exp.log_prob(self.transfs_f[i](x))]
			else:
				if hasattr(exp, '_log_unnormalized_prob'):
					probs += [exp._log_unnormalized_prob(self.transfs_f(x, i))]
				else:
					probs += [exp.log_prob(self.transfs_f(x, i))]

		return probs

	def _log_unnormalized_prob(self, x):
		if x.get_shape().ndims == 1:
			x = x[None]

		cost = 0. if self._cost is None else self._cost(x)
		return tf.reduce_sum(self._experts_probs(x), axis=0) - cost

	@property
	def nb_experts(self):
		return len(self.experts)

