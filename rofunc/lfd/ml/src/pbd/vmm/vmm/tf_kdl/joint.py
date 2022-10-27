import tensorflow as tf
import numpy as np
from .frame import Frame, Twist
from .rotation import *
from enum import IntEnum

class JointType(IntEnum):
	RotX = 0
	RotY = 1
	RotZ = 2
	RotAxis = 3
	NoneT = 4

class Joint(object):
	def	__init__(self, type, origin=None, axis=None, name='', limits=None):
		self.type = type

		if limits is not None:
			self.limits = {'up': limits.upper, 'low': limits.lower,
						   'vel': limits.velocity, 'effort': limits.effort}
		else:
			self.limits = {}

		self.axis = axis
		self.origin = origin



	def pose(self, a):
		# TODO implement origin

		if self.type is JointType.RotX:
			return Frame(m=rot_x(a))
		elif self.type is JointType.RotY:
			return Frame(m=rot_y(a))
		elif self.type is JointType.RotZ:
			return Frame(m=rot_z(a))
		elif self.type is JointType.RotAxis:
			return Frame(p=self.origin , m=rot_2(self.axis, a))
		elif self.type is JointType.NoneT:
			return Frame()

	def twist(self, a):
		if self.type is JointType.RotX:
			return Twist(twist_x(a))
		elif self.type is JointType.RotY:
			return Twist(twist_y(a))
		elif self.type is JointType.RotZ:
			return Twist(twist_z(a))
		elif self.type is JointType.RotAxis:
			return Twist(twist_2(self.axis, a))
		elif self.type is JointType.NoneT:
			return Twist()

class Mesh(object):
	def __init__(self, mesh, dtype=tf.float32):
		self._vertices = tf.convert_to_tensor(mesh.vertices, dtype=dtype)
		self._nb_vertices = mesh.vertices.shape[0]

		self._area_faces = tf.convert_to_tensor(mesh.area_faces, dtype=dtype)

		triangles = np.copy(mesh.triangles)
		# triangles[:, :, 0] = mesh.triangles[:, :, 2]
		# triangles[:, :, 1] = mesh.triangles[:, :, 0]
		# triangles[:, :, 2] = mesh.triangles[:, :, 1]

		self._triangles = tf.convert_to_tensor(triangles, dtype=dtype)
		sample = self.sample(10)

	def sample(self, size=10):
		"""
		Random sample on surface
		:param size:
		:return:
		"""
		# sample triangle proportional to surface
		idx = tf.random.categorical(tf.log(self._area_faces)[None], size)[0]

		triangles_samples = tf.gather(
			self._triangles,
			idx
		)

		# sample on triangle tf.reduce_sum(tf.transpose(vs)[:, :, None] * triangles_samples, axis=1)
		r0, r1 = tf.random_uniform((size, ), 0., 1.), tf.random_uniform((size, ), 0., 1.)

		vs = tf.stack([1. - r0 ** 0.5, r0 ** 0.5 * (1. - r1), r1 * r0 ** 0.5])
		return tf.reduce_sum(tf.transpose(vs)[:, :, None] * triangles_samples, axis=1)

	def sample_face(self, size=10):
		return tf.gather(self._vertices,
			tf.random_uniform((size, ), 0, self._nb_vertices-1, dtype=tf.int64))

class Link(object):
	def __init__(self, frame, mass=1.0):
		self.mass = mass
		self.frame = frame

		self._collision_mesh = None

	@property
	def collision_mesh(self):
		return self._collision_mesh

	@collision_mesh.setter
	def collision_mesh(self, value):
		self._collision_mesh = Mesh(value)

	def pose(self):
		return self.frame