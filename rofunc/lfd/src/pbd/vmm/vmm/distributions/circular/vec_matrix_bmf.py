import tensorflow as tf
from tensorflow_probability import distributions as ds
from ...utils.tf_utils import matmatmul, matvecmul, vec
from ..mvn import MVNFull
from .matrix_bingham_vonmises_fisher import MatrixBinghamVonMisesFisher

class VecMatBMF(ds.Distribution):
	def __init__(self,
				 A=None,
				 B=None,
				 C=None,
				 D=None,
				 loc=None,
				 cov=None,
				 prec=None):

		self._mvn = MVNFull(loc=loc, prec=prec, cov=cov)
		self._bmf = MatrixBinghamVonMisesFisher(A=A, B=B, C=C)

		self._D = D

	def _log_unnormalized_prob(self, x):
		"""

		:param x: [p; vec(R)]
		:return:
		"""

		p = x[:, :3]
		R = tf.reshape(x[:, 3:], (-1, 3, 3))

		# raise NotImplementedError, "Should implement the correlation between position and rotation"
		return self._mvn._log_unnormalized_prob(x) + self._bmf._log_unnormalized_prob(R) +\
			   self._rot_pos_cov(x, R)

	def _rot_pos_cov(self, x, R):
		return tf.einsum('ai,aij->a', x - self._mvn.loc, matmatmul(self.D, R))
		# or return tf.einsum('ai,aij->a', x, matmatmul(self.D, vec(R)))
