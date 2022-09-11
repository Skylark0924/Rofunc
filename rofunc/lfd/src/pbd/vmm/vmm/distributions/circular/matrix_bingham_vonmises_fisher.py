import tensorflow as tf
from tensorflow_probability import distributions as ds
from ...utils.tf_utils import matmatmul, matvecmul, kron, vec
from ..mvn import MVNFull

class MatrixBinghamVonMisesFisher(ds.Distribution):
	def __init__(self,
				 A=None,
				 B=None,
				 C=None,
				 name="MatBinghamMisesFisher"):
		"""
		https://pdfs.semanticscholar.org/a4de/f44bd744e4bb045206254966fac1e2b14b0e.pdf

		:param A:  	symmetric matrix
		:param B:	diagonal matrix
		:param C:
		:param name:
		"""
		self._A = A
		self._B = B
		self._C = C

		self.ndim = A.shape[0].value

		self._name = name
		self._mvn = None

	@property
	def mvn(self):
		"""
		If A is positive defined and B is negative, then the Hessian is negative definite
		and the distribution can be expressed as a Multivariate normal distribution on
		vectorized X.
		:return:
		"""
		if self._mvn is None:
			self._mvn = self.to_vec_mvn()

		return self._mvn

	@property
	def loc(self):
		return self.mvn.loc

	@property
	def prec(self):
		return self.mvn.prec

	@property
	def cov(self):
		return self.mvn.cov

	def to_vec_mvn(self):
		"""
		If A is positive defined and B is negative, then the Hessian is negative definite
		and the distribution can be expressed as a Multivariate normal distribution on
		vectorized X.
		see MBvMF to vec MVN notebook
		"""

		loc = -0.5 * tf.reshape(tf.matrix_transpose(
			matmatmul(matmatmul(tf.linalg.inv(self.B), self.C, transpose_b=True),
					  tf.linalg.inv(self.A))), (-1,))  # set gradient to 0

		prec = -2 * kron(self.A, self.B)

		return MVNFull(loc=loc, prec=prec, name='vec%s' % self.name)

	@property
	def A(self):
		return self._A

	@property
	def B(self):
		return self._B

	@property
	def C(self):
		return self._C

	def _log_unnormalized_prob(self, x):
		"""

		:param x:	[batch_shape, ndim, ndim]
		:type x: 	tf.Tensor
		:return:	[batch_shape]
		"""

		if x.shape.ndims == 2:
			x = tf.reshape(x, (-1, self.ndim, self.ndim))

		return tf.linalg.trace(
			matmatmul(self.C, x, transpose_a=True) +
			matmatmul(self.B, matmatmul(x, matmatmul(self.A, x), transpose_a=True))
		)
