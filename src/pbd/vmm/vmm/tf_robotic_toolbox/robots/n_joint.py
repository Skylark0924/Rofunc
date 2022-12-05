import tensorflow as tf
from .robot import Robot

class NJointRobot(Robot):
	def __init__(self, n=3):
		Robot.__init__(self)

		self._ls = tf.constant(1./float(n) * tf.ones(n))

		pi = 3.14159
		margin = 0.02
		self._joint_limits = tf.constant([
			[0. + margin, pi - margin]] +
			[[-pi/2. + margin, pi/2. - margin]] * (n - 1), dtype=tf.float32)


	def xs(self, q):
		if q.shape.ndims == 1:
			q_currents = tf.cumsum(q)

			x = tf.cumsum(self._ls * tf.cos(q_currents))
			x = tf.concat([tf.zeros(1), x], 0)
			y = tf.cumsum(self._ls * tf.sin(q_currents))
			y = tf.concat([tf.zeros(1), y], 0)

			return tf.transpose(tf.stack([x, y]))


		else:
			q_currents = tf.cumsum(q, axis=1)
			x = tf.cumsum(self._ls[None] * tf.cos(q_currents), axis=1)
			x = tf.concat([tf.zeros_like(x[:, 0][:, None]), x], axis=1)
			y = tf.cumsum(self._ls[None] * tf.sin(q_currents), axis=1)
			y = tf.concat([tf.zeros_like(y[:, 0][:, None]), y], axis=1)

			return tf.concat([x[..., None], y[..., None]], axis=-1)

	def jacobian(self, q):
		if q.shape.ndims == 1:
			X = self.xs(q)[-1]
			dX = tf.reshape(tf.convert_to_tensor([tf.gradients(Xi, q) for Xi in tf.unstack(X)]), (2, self.state.n))
			return dX
		else:
			list_dX = []
			for q_single in tf.unstack(q, axis=0):
				X = self.xs(q_single)[-1]
				dX = tf.reshape(tf.convert_to_tensor([tf.gradients(Xi, q_single) for Xi in tf.unstack(X)]), (2, self.state.n))
				list_dX.append(dX)
			dX = tf.stack(list_dX)
			return dX


