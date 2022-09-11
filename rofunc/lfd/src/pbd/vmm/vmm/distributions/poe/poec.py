import tensorflow as tf
from .poe import PoE

class PoEC(PoE):
	"""
	Product of Conditional experts
	"""
	def __init__(self, shape, experts, transfs_x, transfs_y):
		self._transfs_y = transfs_y
		self._transfs_x = transfs_x

		self._x_lin = tf.placeholder(tf.float32, (None, shape[0]))
		self._y_lin = tf.placeholder(tf.float32, (None, shape[0]))

		# self.jx_x = [jacobians(tr(self.x_lin, self.y_lin), self.x_lin) for tr in transfs_x]
		# self.jx_y = [jacobians(tr(self.x_lin, self.y_lin), self.y_lin) for tr in transfs_x]

		# self.jy_x = [jacobians(tr(self.y_lin, self.x_lin), self.x_lin) for tr in transfs_y]
		self.jy_y = [jacobians(tr(self.y_lin, self.x_lin), self.y_lin) for tr in transfs_y]

		super(PoEC, self).__init__(shape, experts, None)


	@property
	def precs(self):
		"""
		Computes J^T \Lambda J
		:return:
		"""
		assert all([hasattr(exp, 'prec') for exp in self.experts])

		return [tf.einsum('aji,ajk->aik', self.jy_y[i],
						  tf.einsum('ij,ajk->aik', exp.prec, self.jy_y[i]))
				for i, exp in enumerate(self.experts)]

	@property
	def precsT(self):
		"""
		Computes J^T \Lambda
		:return:
		"""
		assert all([hasattr(exp, 'prec') for exp in self.experts])

		return [tf.einsum('aji,jk->aik', self.jy_y[i], exp.prec)
				for i, exp in enumerate(self.experts)]
	@property
	def prec(self):
		return tf.reduce_sum(self.precs, axis=0)

	def musT(self, x):
		"""
		Computes mu - f(y_lin) - J(y_lin) y_lin
		:return:
		"""
		return [exp.loc_pred(tr_x(x, None)) - tr_y(self.y_lin, x) + tf.einsum('aij,aj->ai', self.jy_y[i], self.y_lin)
				for i, (exp, tr_x, tr_y) in enumerate(zip(self.experts, self.transfs_x, self.transfs_y))]

	def mus(self, x):
		return [exp.loc_pred(tr_x(x, None))
			 for exp, tr_x in zip(self.experts, self.transfs_x)]

	def mu(self, x):
		musT = self.musT(x)

		mu = tf.reduce_sum([tf.einsum('aij,aj->ai', _precT, _muT, name='mul_lmbdaT_muT')
							for _muT, _precT in zip(musT, self.precsT)], 0)

		return tf.einsum('bjk,bj->bk', tf.linalg.inv(self.prec), mu)


	@property
	def x_lin(self):
		return self._x_lin

	@property
	def y_lin(self):
		return self._y_lin

	@property
	def transfs_x(self):
		return self._transfs_x

	@property
	def transfs_y(self):
		return self._transfs_y

	def log_prob(self, y, x):
		return self._log_prob(y, x)

	def _log_prob(self, y, x):
		return self._log_unnormalized_prob(y, x) - self._log_normalization()

	def _log_unnormalized_prob(self, y, x):
		return tf.reduce_sum(
			[exp.log_prob(tr_y(y, x), tr_x(x, y))
			 for exp, tr_y, tr_x in zip(self.experts, self.transfs_y, self.transfs_x)], axis=0)

