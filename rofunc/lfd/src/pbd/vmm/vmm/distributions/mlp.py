import tensorflow as tf
from tensorflow_probability import distributions as ds
from ..utils.tf_utils import matvecmul, matmatmul, tau
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
import numpy as np

class MLP(object):
	def __init__(self, n_input, n_output, n_hidden=[12, 12], batch_size_svi=10, act_fct=tf.nn.relu):

		self._act_fct = act_fct
		self._batch_size_svi = batch_size_svi
		self._weights_size = 0
		self._layer_size = len(n_hidden) + 1

		shapes = np.array([n_input] + n_hidden + [n_output])

		self._n_input = n_input
		self._n_hidden = n_hidden
		self._n_output = n_output
		self._shapes = shapes

		# add weights n_input * n_hidden_1 + n_hidden_1 * n_hidden_2 + n_hidden_2 * n_output
		self._weights_size += np.sum(shapes[:-1] * shapes[1:])

		# add biased
		self._weights_size += np.sum(shapes[1:])

	@property
	def weights_size(self):
		return self._weights_size

	@property
	def weights_shape(self):
		return [self._batch_size_svi, self._weights_size]

	def unpack_weights(self, vec_weights):
		idx_s = 0
		idx_layer = 0

		weights = {}

		for s0, s1 in zip(self._shapes[:-1], self._shapes[1:]):
			idx_e = idx_s + s0 * s1

			weights['h%d' % idx_layer] = tf.reshape(vec_weights[:, idx_s:idx_e], (-1, s0, s1))

			idx_layer += 1
			idx_s = idx_e


		idx_layer = 0

		biases = {}

		for s0 in self._shapes[1:]:
			idx_e = idx_s + s0

			biases['b%d' % idx_layer] = vec_weights[:, idx_s:idx_e]

			idx_layer += 1
			idx_s = idx_e

		return weights, biases

	def multilayer_perceptron(self, x, vec_weights):
		"""
		:param x: [batch_size, n_input]
		:return [batch_size_svi, batch_size, n_output]
		"""
		weights, biases = self.unpack_weights(vec_weights)
		batch_size = tf.shape(x)[0]
		layer = tf.tile(tf.expand_dims(tf.identity(x),0),[self._batch_size_svi,1,1])

		for i in range(self._layer_size):
			bias = tf.tile(tf.expand_dims(biases['b%d' % i], 1), [1, batch_size, 1])
			if i == self._layer_size - 1:
				layer = tf.add(tf.einsum('aik, akj-> aij', layer, weights['h%d' % i]), bias)
			else:
				layer = self._act_fct(tf.add(tf.einsum('aik, akj-> aij', layer, weights['h%d' % i]), bias))

		return layer

	def RNN(self, x, vec_weights):

		weights, biases = self.unpack_weights(vec_weights)

		# Prepare data shape to match `rnn` function requirements
		# Current data input shape: (batch_size, timesteps, n_input)
		# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

		batch_size = tf.shape(x)[0]
		timesteps = tf.shape(x)[1]
		# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
		x = tf.unstack(x, axis=1)
		i = 1
		bias = tf.tile(tf.expand_dims(biases['b%d' % i], 1), [1, batch_size, 1])
		bias = tf.tile(tf.expand_dims(bias, 1), [1, timesteps, 1, 1])

		# Define a lstm cell with tensorflow
		lstm_cell = tf.contrib.rnn.LSTMBlockCell(self._n_hidden[0], forget_bias=1.0)
		# Get lstm cell output
		outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
		outputs = tf.stack(outputs)
		# Linear activation, using rnn inner loop last output
		layer = tf.add(tf.einsum('tbh, ahj-> atbj', outputs, weights['h%d' % i]), bias)

		return layer

