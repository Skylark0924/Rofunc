import tensorflow as tf
from .robot import Robot

class ThreeJointRobot(Robot):
	def __init__(self, ls=None):
		Robot.__init__(self)

		L = 0.25
		self._ls = tf.constant([L, L, L]) if ls is None else ls

		pi = 3.14159
		margin = 0.02

		self._joint_limits = tf.constant([[0. + margin, pi - margin],
										  [-pi/2. + margin, pi/2. - margin],
										  [-pi/2. + margin, pi/2. - margin]],
										 dtype=tf.float32)


	def xs(self, q, x_base=None, angle=False):
		"""

		:param q:
		:param x_base: [2] or [batch_size, 2]
		:param angle:
		:return:
		"""
		if q.shape.ndims == 1:
			x = tf.cumsum([0,
						   self.ls[0] * tf.cos(q[0]),
						   self.ls[1] * tf.cos(q[0] + q[1]),
						   self.ls[2] * tf.cos(q[0] + q[1] + q[2])])
			y = tf.cumsum([0,
						   self.ls[0] * tf.sin(q[0]),
						   self.ls[1] * tf.sin(q[0] + q[1]),
						   self.ls[2] * tf.sin(q[0] + q[1] + q[2])])

			if x_base is not None:
				x += x_base[0]
				y += x_base[1]

			if angle:
				return tf.transpose(tf.stack([x, y, tf.cumsum(tf.concat([q, tf.zeros_like(q[0][None])], axis=0))]))
			else:
				return tf.transpose(tf.stack([x, y]))
		else:
			x = tf.cumsum([tf.zeros_like(q[:, 0]),
						   self.ls[None, 0] * tf.cos(q[:, 0]),
						   self.ls[None, 1] * tf.cos(q[:, 0] + q[:, 1]),
						   self.ls[None, 2] * tf.cos(q[:, 0] + q[:, 1] + q[:, 2])])

			y = tf.cumsum([tf.zeros_like(q[:, 0]),
						   self.ls[None, 0] * tf.sin(q[:, 0]),
						   self.ls[None, 1] * tf.sin(q[:, 0] + q[:, 1]),
						   self.ls[None, 2] * tf.sin(q[:, 0] + q[:, 1] + q[:, 2])])

			if x_base is not None:
				x += x_base[:, 0]
				y += x_base[:, 1]

			if angle:
				return tf.transpose(tf.stack([x, y, tf.cumsum(tf.concat([tf.transpose(q), tf.zeros_like(q[:, 0][None])], axis=0))]), (2, 1, 0))
			else:
				return tf.transpose(tf.stack([x, y]), (2, 1, 0))

