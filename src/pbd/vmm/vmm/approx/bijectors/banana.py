import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

class BananaNd(tfp.bijectors.Bijector):
	def __init__(self, kappa, dim=0, name="banana"):
		self.kappa = kappa

		self.d = kappa.shape[-1].value
		_mask = np.ones(self.d, dtype=np.float32)
		_mask[dim] = 0.

		self.mask_get = tf.constant(_mask)
		_mask = np.zeros(self.d, dtype=np.float32)
		_mask[dim] = 1.
		self.mask_set = tf.constant(_mask)

		super(BananaNd, self).__init__(
			inverse_min_event_ndims=1, name=name, is_constant_jacobian=True)

	def _forward(self, x):
		y = x - self.mask_set * tf.reduce_sum(self.mask_get * self.kappa * (x ** 2),
											  axis=-1, keepdims=True)
		return y

	def _inverse(self, y):
		x = y + self.mask_set * tf.reduce_sum(self.mask_get * self.kappa * (y ** 2),
											  axis=-1, keepdims=True)
		return x

	def _inverse_log_det_jacobian(self, y):
		return tf.zeros(shape=())