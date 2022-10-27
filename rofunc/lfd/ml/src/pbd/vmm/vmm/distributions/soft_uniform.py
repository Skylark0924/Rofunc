import tensorflow as tf
from tensorflow_probability import distributions as ds
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops


class SoftUniform(ds.Distribution):
	def __init__(self,
				 low=0.,
				 high=1.,
				 temp=1e2,
				 validate_args=False,
				 allow_nan_stats=True,
				 squared=True,
				 activation=tf.nn.relu,
				 name="SoftUniform"):
		parameters = locals()

		parameters = dict(locals())

		self._activation = activation
		self._squared = squared

		with ops.name_scope(name, values=[low, high]) as name:
			with ops.control_dependencies([
											  check_ops.assert_less(
												  low, high,
												  message="uniform not defined when low >= high.")
										  ] if validate_args else []):
				self._low = array_ops.identity(low, name="low")
				self._high = array_ops.identity(high, name="high")
				self._temp = array_ops.identity(temp, name="temp")
				check_ops.assert_same_float_dtype([self._low, self._high, self._temp])

		super(SoftUniform, self).__init__(
			dtype=self._low.dtype,
			reparameterization_type=ds.FULLY_REPARAMETERIZED,
			validate_args=validate_args,
			allow_nan_stats=allow_nan_stats,
			parameters=parameters,
			graph_parents=[self._low,
						   self._high,
						   self._temp],
			name=name)

	@property
	def event_shape(self):
		return self._low.shape	

	@property
	def temp(self):
		return self._temp

	def log_prob(self, value, name="log_prob"):
		z = self._high - self._low

		log_unnormalized_prob = 0.

		if self._squared:
			if self._high.shape.ndims == 0:
				log_unnormalized_prob += (self._temp * self._activation(value - self._high)) ** 2
				log_unnormalized_prob += (self._temp * self._activation(-value + self._low)) ** 2
			else:
				z = tf.reduce_prod(z)
				log_unnormalized_prob += tf.reduce_sum((self._temp * self._activation(-value + self._low[None])) ** 2, -1)
				log_unnormalized_prob += tf.reduce_sum((self._temp * self._activation(value - self._high[None])) ** 2, -1)
		else:
			if self._high.shape.ndims == 0:
				log_unnormalized_prob += self._temp * self._activation(value - self._high)
				log_unnormalized_prob += self._temp * self._activation(-value + self._low)
			else:
				z = tf.reduce_prod(z)
				log_unnormalized_prob += tf.reduce_sum(self._temp * self._activation(value - self._high[None]), -1)
				log_unnormalized_prob += tf.reduce_sum(self._temp * self._activation(-value + self._low[None]), -1)

		return -tf.log(z) - log_unnormalized_prob

class SoftUniformNormalCdf(SoftUniform):
	"""
	SoftUniform with normal log cdf
	"""
	def __init__(self,
				 low=0.,
				 high=1.,
				 std=0.1,
				 temp=1.,
				 reduce_axis=None,
				 validate_args=False,
				 allow_nan_stats=True,
				 name="SoftUniformNormalCdf"):
		"""

		:param low:
		:param high:
		:param std:
		:param temp:
		:param reduce_axis:  if not None, on which axis to reduce log prob, for independant constraints
		:param validate_args:
		:param allow_nan_stats:
		:param name:
		"""
		with ops.name_scope(name, values=[low, high]) as name:
			with ops.control_dependencies([
											  check_ops.assert_less(
												  low, high,
												  message="uniform not defined when low >= high.")
										  ] if validate_args else []):
				self._std = array_ops.identity(std, name="std")

		self._reduce_axis = reduce_axis
		super(SoftUniformNormalCdf, self).__init__(
			low=low,
			high=high,
			temp=temp,
			validate_args=validate_args,
			allow_nan_stats=allow_nan_stats,
			name=name)

	def log_prob(self, value, name="log_prob"):
		log_probs = self._temp * (ds.Normal(self._low, self._std).log_cdf(value) +
			   ds.Normal(-self._high, self._std).log_cdf(-value))

		if self._reduce_axis is not None:
			return tf.reduce_sum(log_probs, axis=self._reduce_axis)
		else:
			return log_probs
