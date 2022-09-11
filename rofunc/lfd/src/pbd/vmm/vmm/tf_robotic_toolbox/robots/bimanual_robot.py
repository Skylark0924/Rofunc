import tensorflow as tf
from .robot import Robot
from .three_joint import ThreeJointRobot

class BimanualThreeJointRobot(Robot):
	def __init__(self):
		Robot.__init__(self)

		L = 0.25
		self._ls = tf.constant([L, L, L, L, L])

		self._arms = [
			ThreeJointRobot(),
			ThreeJointRobot(),
		]

	def joint_limit_cost(self, q, std=0.1):
		qs = [q[:, 0:3], tf.concat([q[:, 0][:, None], q[:, 3:5]], axis=1)]
		return self._arms[0].joint_limit_cost(qs[0], std=std) + self._arms[0].joint_limit_cost(qs[1], std=std)

	def xs(self, q, concat=True):
		if q.shape.ndims == 1:
			qs = [q[0:3], tf.concat([q[0][None], q[3:5]], axis=0)]
			return tf.concat([self._arms[0].xs(qs[0]), self._arms[1].xs(qs[0])], axis=0)

		else:
			qs = [q[:, 0:3], tf.concat([q[:, 0][:, None], q[:, 3:5]], axis=1)]

			fks = [
				self._arms[0].xs(qs[0]),
				self._arms[1].xs(qs[1])
			]
			if concat:
				return tf.concat(fks, axis=1)
			else:
				return fks

		return