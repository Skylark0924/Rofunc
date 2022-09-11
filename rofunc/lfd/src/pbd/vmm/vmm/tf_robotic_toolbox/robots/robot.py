import tensorflow as tf
import numpy as np

from tensorflow_probability import distributions as ds


class Robot(object):
	def __init__(self):
		self._ls, self._ms, self._ins = [], [], []

		self._joint_limits = tf.constant([
			[0., 0.],
		], dtype=tf.float32)

		self._base_limits = tf.constant([[-1., 1.],
										 [-1., 1.]],
										dtype=tf.float32)

	def base_limit_cost(self, x, std=0.1, base_limit=1.):
		return -ds.Normal(base_limit * self._base_limits[:, 0], std).log_cdf(x) - ds.Normal(
			-base_limit * self._base_limits[:, 1], std).log_cdf(-x)

	def joint_limit_cost(self, q, std=0.1):
		return -ds.Normal(self._joint_limits[:, 0], std).log_cdf(q) - ds.Normal(
			-self._joint_limits[:, 1], std).log_cdf(-q)

	def segment_samples(self, q, nsamples_segment=10, noise_scale=None):

		if q.shape.ndims == 2:
			segments = self.forward_kin(q)  # batch_shape, n_points, 2

			n_segments = segments.shape[1].value - 1
			samples = []

			for i in range(n_segments):
				u = tf.random_uniform((nsamples_segment, ))

				# linear interpolations between end-points of segments
				#  batch_shape, nsamples_segment, 2
				samples += [u[None, :, None] * segments[:, i][:, None]
							+ (1.- u[None, :, None]) * segments[:, i+1][:, None]]


				if noise_scale is not None:
					samples[-1] += tf.random_normal(samples[-1].shape, 0., noise_scale)

			return tf.concat(samples, axis=1)

		else:
			raise NotImplementedError

	def min_sq_dist_from_point(self, q, x, **kwargs):
		"""

		:param q: [batch_shape, 2]
		:param x: [2, ]
		:return:
		"""
		if q.shape.ndims == 2:
			samples = self.segment_samples(q, **kwargs) # batch_shape, nsamples, 2

			dist = tf.reduce_sum((samples - x[None, None])**2, axis=2)

			return tf.reduce_min(dist, axis=1)

	def xs(self, q):
		raise NotImplementedError

	def jacobian(self, q):
		raise NotImplementedError

	@property
	def ls(self):
		return self._ls
