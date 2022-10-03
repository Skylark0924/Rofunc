import tensorflow as tf
from tensorflow_probability import distributions as ds
from ..utils.tf_utils import matvecmul, matmatmul, tau
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops

class MVNFull(ds.Distribution):
	def __init__(self,
				 loc=None,
				 prec=None,
				 cov=None,
				 reg=None,
				 is_prec=True,
				 n_dim=None,
				 validate_args=False,
				 allow_nan_stats=True,
				 name="MultivariateNormalFull"):
		parameters = locals()

		self._prec = prec
		self._cov = cov
		self._loc = loc

		self._is_prec = is_prec and prec is not None

		self.reg = reg
		dtype = self._loc.dtype if self._loc is not None else self._eta.dtype

		super(MVNFull, self).__init__(
			dtype=dtype,
			reparameterization_type=ds.FULLY_REPARAMETERIZED,
			validate_args=validate_args,
			allow_nan_stats=allow_nan_stats,
			parameters=parameters,
			# graph_parents=[self._loc, self._prec],
			name=name)

	@property
	def is_prec(self):
		"""
		Return True if precision matrix is main parametrization - return false if covariance
		is the main.
		:return:
		"""
		return self._is_prec

	def _sample_n(self, n, seed=None):
		# TODO implement sampling
		shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
		sampled = random_ops.random_normal(
			shape=shape, mean=0., stddev=1., dtype=self.experts[0].loc.dtype, seed=seed)
		return tf.einsum('ij,aj->ai', tf.linalg.cholesky(self.cov),  sampled) + self.loc

	def _batch_shape_tensor(self):
		return array_ops.shape(self.loc)

	def _batch_shape(self):
		if self._loc is not None:
			return array_ops.broadcast_static_shape(
				self._loc.get_shape()[:-1],
				self.prec.get_shape()[:-2])
		else:
			return array_ops.broadcast_static_shape(
				self._eta.get_shape()[:-1],
				self.prec.get_shape()[:-2])

	def _event_shape_tensor(self):
		return constant_op.constant([], dtype=dtypes.int32)

	def _event_shape(self):
		if self._loc is not None:
			return self._loc.get_shape()[-1:]
		else:
			return self._eta.get_shape()[-1:]

	def _sample_n(self, n, seed=None):
		"""

		:param n:
		:param seed:
		:return: 		[[n] + [batch_shape] + [event_shape]]
		"""
		shape = array_ops.concat([[n], self.batch_shape_tensor(), self.event_shape_tensor()], 0)
		sampled = random_ops.random_normal(
			shape=shape, mean=0., stddev=1., dtype=self.loc.dtype, seed=seed)
		# if reg is None:
		# 	return tf.einsum('ij,aj->ai', tf.linalg.cholesky(self.cov),  sampled) + self.loc
		# else:
		return tf.einsum('ij,aj->ai', tf.linalg.cholesky(self.cov),  sampled) + self.loc


	@property
	def loc(self):
		"""Distribution parameter for the mean."""
		return self._loc

	@property
	def prec(self):
		"""Distribution parameter for the precision matrix."""
		if self._prec is None:
			self._prec = tf.linalg.inv(self._cov, name='%s_InvPrec' % self.name)

		return self._prec

	@property
	def cov(self):
		if self._cov is None:
			self._cov = tf.linalg.inv(self._prec, name='%s_InvPrec' % self.name)

		return self._cov

	@property
	def eta(self):
		if self._eta is None:
			self._eta = matvecmul(self.prec, self.loc)

		return self._eta


	@property
	def is_valid(self):
		if self.is_valid:
			assert self.loc is not None

		return self._is_valid

	def _log_unnormalized_prob(self, x):
		dx = self.loc - x
		if dx.shape.ndims == 1:
			return -0.5 * tf.reduce_sum(dx * tf.einsum('jk,j->k', self.prec, dx))
		elif dx.shape.ndims == 2 and self.prec.shape.ndims == 2:
			return -0.5 * tf.reduce_sum(dx * tf.einsum('jk,aj->ak', self.prec, dx), axis=1)
		elif dx.shape.ndims == 2 and self.prec.shape.ndims == 3:
			return -0.5 * tf.reduce_sum(dx * tf.einsum('ajk,aj->ak', self.prec, dx), axis=1)

	def _log_normalization(self):
		if self.is_prec:
			return -0.5 * tf.linalg.logdet(tau * self.prec)
		else:
			return 0.5 * tf.linalg.logdet(tau * self.cov)

	def _log_prob(self, x):
		return self._log_unnormalized_prob(x) - self._log_normalization()