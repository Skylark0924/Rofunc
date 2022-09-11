import numpy as np
import tensorflow as tf
from scipy.special import logsumexp
def vec(x):
	"""

	:param x:	[batch_shape, d1, d2]
	:return:  [batch_shape, d1 * d2]
	"""
	d1, d2 = [s.value for s in  x.shape[1:]]

	return tf.reshape(x, (-1, d1 * d2))


def log_normalize(x, axis=0):
	return x - logsumexp(x, axis=axis)


def matmatmul(a=None, b=None, transpose_a=False, transpose_b=False):
	"""

	:param a: ...ij
	:param b: ...jk
	:param transpose_a:
	:param transpose_b:
	:return: ...ik
	"""
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
