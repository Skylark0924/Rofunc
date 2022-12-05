import tensorflow as tf
from tensorflow_probability import distributions as ds
from ...utils.tf_utils import matmatmul

class VonMisesFisher(ds.Distribution):
	def __init__(self,
				 F=None,
				 name="VonMises_Fisher"):

		"""

		:param F: 	[n_dim, n_dim]
		:param name:
		"""

		self._F = F

		self._name = name

	@property
	def F(self):
		return self._F

	def _log_unnormalized_prob(self, x):
		"""

		:param x:	[batch_shape, ndim, ndim]
		:type x: 	tf.Tensor
		:return:	[batch_shape]
		"""

		return tf.linalg.trace(matmatmul(self.F, x, transpose_a=True))



