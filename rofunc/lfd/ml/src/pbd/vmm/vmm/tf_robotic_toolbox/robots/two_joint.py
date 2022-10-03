import tensorflow as tf
from .robot import Robot


class TwoJointRobot(Robot):
	def __init__(self, ls=None):
		Robot.__init__(self)

		L = 0.25
		self._ls = tf.constant([L, L]) if ls is None else ls

		margin = 0.02
		pi = 3.14159

		self._joint_limits = tf.constant([
			[0. + margin, pi - margin],
			[-pi + margin, pi - margin],
		], dtype=tf.float32)


	def jacobian(self, q):
		if q.shape.ndims == 1:
			J = [[0. for i in range(2)] for j in range(2)]
			J[0][1] = self.ls[1] * -tf.sin(q[0] + q[1])
			J[1][1] = self.ls[1] * tf.cos(q[0] + q[1])
			J[0][0] = self.ls[0] * -tf.sin(q[0]) + J[0][1]
			J[1][0] = self.ls[0] * tf.cos(q[0]) + J[1][1]

			arr = tf.stack(J)
			return tf.reshape(arr, (2, 2))
		else:
			J = [[0. for i in range(2)] for j in range(2)]
			J[0][1] = self.ls[1] * -tf.sin(q[:, 0] + q[:, 1])
			J[1][1] = self.ls[1] * tf.cos(q[:, 0] + q[:, 1])
			J[0][0] = self.ls[0] * -tf.sin(q[:, 0]) + J[0][1]
			J[1][0] = self.ls[0] * tf.cos(q[:, 0]) + J[1][1]

			arr = tf.stack(J)
			return tf.transpose(arr, (2, 0, 1))


	def xs(self, q):
		if q.shape.ndims == 1:
			x = tf.cumsum([0,
						   self.ls[0] * tf.cos(q[0]),
						   self.ls[1] * tf.cos(q[0] + q[1])])
			y = tf.cumsum([0,
						   self.ls[0] * tf.sin(q[0]),
						   self.ls[1] * tf.sin(q[0] + q[1])])

			return tf.transpose(tf.stack([x, y]))

		else:
			x = tf.cumsum([tf.zeros_like(q[:, 0]),
						   self.ls[None, 0] * tf.cos(q[:, 0]),
						   self.ls[None, 1] * tf.cos(q[:, 0] + q[:, 1])])
			y = tf.cumsum([tf.zeros_like(q[:, 0]),
						   self.ls[None, 0] * tf.sin(q[:, 0]),
						   self.ls[None, 1] * tf.sin(q[:, 0] + q[: ,1])])

			return tf.transpose(tf.stack([x, y]), (2, 1, 0))
