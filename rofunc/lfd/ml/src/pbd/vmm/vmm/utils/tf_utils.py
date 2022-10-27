import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops, control_flow_ops, tensor_array_ops
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian


tau = 6.283185307179586


def damped_pinv_right(J, d=1e-5):
	"""Minimizing x velocity"""

	s = J.shape[-1] if isinstance(J.shape[-1], int) else J.shape[-1].value

	return tf.matmul(tf.linalg.inv(tf.matmul(J, J, transpose_a=True) + d * tf.eye(s)), J,
					 transpose_b=True)

def nullspace_transformation(x, f, d=1e-5):
	"""
	Get nullspace filter of a transformation where dim_in > dim_out

	:param q: 	Tensor [batch_size, dim_in]
	:param f:	function [batch_size, dim_in] -> [batch_size, dim_out]
	:return:	nullspace [batch_size, dim_in, dim_in]
	"""
	y = f(x)
	J = batch_jacobians(y, x)

	s = x.shape[-1] if isinstance(x.shape[-1], int) else x.shape[-1].value

	return tf.eye(s)[None] - tf.matmul(damped_pinv_right(J, d=d), J)


def nullspace_project(fct, f_main, x):
	"""
	Project gradient of f in nullspace of f_main, return y and gradient
	for redefining gradient

	example of usage :

		@tf.custom_gradient
		def f1_filtered(q):
			return nullspace_project(f1, f0, x)


	:param f:		function [batch_size, dim_in] -> [batch_size, dim_out_1]
	:param f_main :	function [batch_size, dim_in] -> [batch_size, dim_out_2]
	:param x:		Tensor [batch_size, dim_in]
	:return:
	"""
	def grad(dx):
		J = batch_jacobians(fct(x), x)
		J = tf.einsum(
			'aij,ajk->aik', J, tf.stop_gradient(
				nullspace_transformation(x, lambda x: f_main(x))))

		return tf.einsum('ai,aij->aj', dx, J)

	return fct(x), grad

def log_normalize(x, axis=0):
	return x - tf.reduce_logsumexp(x, axis=axis)

def discretize_log_prob(log_prob, nb_sub=40, xlim=[-1., 1.], ylim=[-1., 1.], normalize=False):
	"""
	Evaluate log_prob on a 2D grid
	:param log_prob:
	:param nb_sub:
	:param xlim:
	:param ylim:
	:param normalize:
	:return:
	"""
	x, y = tf.meshgrid(
		tf.linspace(xlim[0], xlim[1], nb_sub), tf.linspace(ylim[0], ylim[1], nb_sub))
	xs = tf.concat([tf.reshape(x, (-1 , 1)), tf.reshape(y, (-1, 1))], axis=-1)
	zs = tf.reshape(log_prob(xs), (nb_sub, nb_sub))
	if normalize:
		zs = log_normalize(zs, axis=(0, 1))
	return zs

def discretized_bhattacharyya_coef(log_0, log_1, nb_sub=40, xlim=[-1., 1.], ylim=[-1., 1.]):
	return tf.reduce_sum((
		tf.exp(discretize_log_prob(log_0, xlim=xlim, ylim=ylim, nb_sub=nb_sub, normalize=True)) *
		tf.exp(discretize_log_prob(log_1, xlim=xlim, ylim=ylim, nb_sub=nb_sub, normalize=True))
	) ** 0.5)

def discretized_alpha12_divergence(log_0, log_1, nb_sub=40, xlim=[-1., 1.], ylim=[-1., 1.]):
	return tf.reduce_sum((
		tf.exp(discretize_log_prob(log_0, xlim=xlim, ylim=ylim, nb_sub=nb_sub, normalize=True)) ** 0.5 -
		tf.exp(discretize_log_prob(log_1, xlim=xlim, ylim=ylim, nb_sub=nb_sub, normalize=True)) ** 0.5
	) ** 2)

def has_parent_variable(tensor, variable):
	"""
	Check if tensor depends on variable
	:param tensor:
	:param variable:
	:return:
	"""
	var_list = len([var for var in tensor.op.values() if var == variable._variable])

	if var_list:
		return True

	for i in tensor.op.inputs:
		if has_parent_variable(i, variable):
			return True

	return False

def get_parent_variables(tensor, sess=None):
	"""
	Get all variables in the graph on which the tensor depends
	:param tensor:
	:param sess:
	:return:
	"""
	if sess is None:
		sess = tf.get_default_session()

	if isinstance(tensor, tf.Variable):
		return [tensor]

	return [v for v in sess.graph._collections['variables'] if
			has_parent_variable(tensor, v)]

def kron(a, b):
	return tf.contrib.kfac.utils.kronecker_product(a, b)

def batch_jacobians(ys, xs):
	"""
	ys : [None, n_y] or [n_y]
	xs : [None, n_x] or [n_x]
	"""
	if ys.shape.ndims == 2:
		s = ys.shape[-1] if isinstance(ys.shape[-1], int) else ys.shape[-1].value
		return tf.transpose(
			tf.stack([tf.gradients(ys[:, i], xs)[0] for i in range(s)]),
			(1, 0, 2))
	elif ys.shape.ndims == 1:
		s = ys.shape[0] if isinstance(ys.shape[0], int) else ys.shape[0].value
		return tf.stack([tf.gradients(ys[i], xs)[0] for i in range(s)])
	else:
		raise NotImplementedError

def jacobians(ys, xs):
	"""
	ys : [None, n_y] or [n_y]
	xs : [None, n_x] or [n_x]
	return [None, n_y, n_x]
	"""

	if ys.shape.ndims == 2:
		return tf.transpose(
			tf.stack([tf.gradients(ys[:, i], xs)[0] for i in range(ys.shape[-1].value)]),
			(1, 0, 2))
	elif ys.shape.ndims == 1:
		return tf.stack([tf.gradients(ys[i], xs)[0] for i in range(ys.shape[0].value)])
	else:
		raise NotImplementedError

def vec(x):
	"""

	:param x:	[batch_shape, d1, d2]
	:return:  [batch_shape, d1 * d2]
	"""
	d1, d2 = [s.value for s in  x.shape[1:]]

	return tf.reshape(x, (-1, d1 * d2))

def matmatmul(a=None, b=None, transpose_a=False, transpose_b=False):
	if a is None:
		return b
	if b is None:
		return a

	if a.shape.ndims == b.shape.ndims:
		return tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
	else:
		idx_a = 'ij,' if not transpose_a else 'ji,'
		idx_b = 'jk->' if not transpose_b else 'kj->'
		idx_c = 'ik'
		b_a = ['', 'a', 'ab', 'abc'][a.shape.ndims-2]
		b_b = ['', 'a', 'ab', 'abc'][b.shape.ndims-2]
		b_c = ['', 'a', 'ab', 'abc'][max([a.shape.ndims, b.shape.ndims])-2]
		return tf.einsum(
			b_a+idx_a+b_b+idx_b+b_c+idx_c, a, b
						 )


def matvecmul(m=None, v=None, transpose_m=False):
	if m is None:
		return v

	if m.shape.ndims == 2 and v.shape.ndims == 1:
		return tf.matmul(m, v[:, None], transpose_a=transpose_m)[:, 0]
	elif m.shape.ndims == 2 and v.shape.ndims == 2:
		return tf.transpose(tf.matmul(m, v, transpose_b=True, transpose_a=transpose_m))
	else:
		idx_a = 'ij,' if not transpose_m else 'ji,'
		idx_b = 'j->'
		idx_c = 'i'

		b_a = ['', 'a', 'ab', 'abc'][m.shape.ndims-2]
		b_b = ['', 'a', 'ab', 'abc'][v.shape.ndims-1]
		b_c = ['', 'a', 'ab', 'abc'][max([m.shape.ndims-2, v.shape.ndims-1])]

		return tf.einsum(
			b_a+idx_a+b_b+idx_b+b_c+idx_c, m, v
		)

def vecvecadd(v1=None, v2=None, opposite_a=False, opposite_b=False):
	if v1 is None:
		return v2

	if v2 is None:
		return v1

	if opposite_a: v1 = -v1
	if opposite_b: v2 = -v2

	if v1.shape.ndims == v2.shape.ndims:
		return v1 + v2
	elif v1.shape.ndims == 2 and v2.shape.ndims == 1 or \
			v1.shape.ndims == 3 and v2.shape.ndims == 2:
		return v1 + v2[None, :]
	elif v1.shape.ndims == 1 and v2.shape.ndims == 2 or \
			v1.shape.ndims == 2 and v2.shape.ndims == 3:
		return v1[None, :] + v2
	else:
		raise NotImplementedError

def block_diagonal(ms):
	"""
	Create a block diagonal matrix with a list of square matrices of same sizes

	:type ms: 		lisf of tf.Tensor	[..., n_dim, n_dim]
	:return:
	"""

	n_dims = np.array([m.shape[-1].value for m in ms])

	if np.sum((np.mean(n_dims) - n_dims) ** 2):  # check if not all same dims
		return block_diagonal_different_sizes(ms)

	s = ms[0].shape[-1].value
	z = ms[0].shape.ndims - 2  # batch dims
	n = len(ms)  # final size of matrix
	mat = []

	for i, m in enumerate(ms):
		nb, na = i * s, (n - i - 1) * s
		paddings = [[0, 0] for i in range(z)] + [[nb, na], [0, 0]]
		mat += [tf.pad(m, paddings=paddings)]

	return tf.concat(mat, -1)

def block_diagonal_different_sizes(ms):

	s = np.array([m.shape[-1].value for m in ms])
	cs = [0] + np.cumsum(s).tolist()
	z = ms[0].shape.ndims - 2  # batch dims
	mat = []

	for i, m in enumerate(ms):
		nb, na = cs[i], cs[-1] - cs[i] - s[i]
		paddings = [[0, 0] for i in range(z)] + [[nb, na], [0, 0]]
		mat += [tf.pad(m, paddings=paddings)]

	return tf.concat(mat, -1)

def _AsList(x):
	return x if isinstance(x, (list, tuple)) else [x]

def batch_hessians(ys,
             xs,
             name="hessians",
             colocate_gradients_with_ops=False,
             gate_gradients=False,
             aggregation_method=None):
	"""

	:param ys: [batch_shape, dim]
	:param xs: [batch_shape, dim]
	:param name:
	:param colocate_gradients_with_ops:
	:param gate_gradients:
	:param aggregation_method:
	:return:
	"""
	xs = _AsList(xs)
	kwargs = {
	  "colocate_gradients_with_ops": colocate_gradients_with_ops,
	  "gate_gradients": gate_gradients,
	  "aggregation_method": aggregation_method
	}
	# Compute first-order derivatives and iterate for each x in xs.
	hessians = []
	_gradients = tf.gradients(ys, xs, **kwargs)

	for gradient, x in zip(_gradients, xs):
		# change shape to one-dimension without graph branching
		# gradient = array_ops.reshape(gradient, [-1])
		# print gradient
		#
		# Declare an iterator and tensor array loop variables for the gradients.
		n = array_ops.shape(x)[-1]

		loop_vars = [
			array_ops.constant(0, dtypes.int32),
			tensor_array_ops.TensorArray(x.dtype, n)
		]
		# Iterate over all elements of the gradient and compute second order
		# derivatives.

		_, hessian = control_flow_ops.while_loop(
			lambda j, _: j < n,
			lambda j, result: (j + 1,
							   result.write(j, tf.gradients(gradient[:, j], x)[0])),
			loop_vars
		)

		# _reshaped_hessian = array_ops.reshape(hessian.stack(),
		# 									  array_ops.concat((_shape, _shape), 0))
		# hessians.append(_reshaped_hessian)
		hessians.append(tf.transpose(hessian.stack(), perm=(1, 0, 2)))

	return hessians

def hessians(ys,
             xs,
             name="hessians",
             colocate_gradients_with_ops=False,
             gate_gradients=False,
             aggregation_method=None):
	xs = _AsList(xs)
	kwargs = {
	  "colocate_gradients_with_ops": colocate_gradients_with_ops,
	  "gate_gradients": gate_gradients,
	  "aggregation_method": aggregation_method
	}
	# Compute first-order derivatives and iterate for each x in xs.
	hessians = []
	_gradients = tf.gradients(ys, xs, **kwargs)
	for gradient, x in zip(_gradients, xs):
		# change shape to one-dimension without graph branching
		gradient = array_ops.reshape(gradient, [-1])

		# Declare an iterator and tensor array loop variables for the gradients.
		n = array_ops.size(x)
		loop_vars = [
			array_ops.constant(0, dtypes.int32),
			tensor_array_ops.TensorArray(x.dtype, n)
		]
		# Iterate over all elements of the gradient and compute second order
		# derivatives.

		_, hessian = control_flow_ops.while_loop(
			lambda j, _: j < n,
			lambda j, result: (j + 1,
							   result.write(j, tf.gradients(gradient[j], x)[0])),
			loop_vars
		)

		_shape = array_ops.shape(x)
		_reshaped_hessian = array_ops.reshape(hessian.stack(),
											  array_ops.concat((_shape, _shape), 0))
		hessians.append(_reshaped_hessian)
	return hessians

