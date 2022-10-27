import tensorflow as tf
from tensorflow_probability import distributions as ds
from ...utils.tf_utils import matmatmul, matvecmul, vec

class FisherBingham(ds.Distribution):
	def __init__(self,
				 A=None,
				 B=None,
				 name="Fisher_Bingham"):

		self._A = A
		self._B = B

		self._name = name

	@property
	def A(self):
		return self._A

	@property
	def B(self):
		return self._B


	def _log_unnormalized_prob(self, x):
		"""

		:param x:	[batch_shape, ndim, ndim]
		:type x: 	tf.Tensor
		:return:	[batch_shape]
		"""

		vec_x = vec(x)

		return tf.linalg.trace(matmatmul(self.A, x, transpose_a=True)) + \
			   tf.reduce_sum(vec_x * matvecmul(self.B, vec_x), axis=1)