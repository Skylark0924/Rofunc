import tensorflow as tf
from .tf_utils import matmatmul

def logm(A):
	# Computes the logm of each matrix in an array containing k positive
	u, v = tf.linalg.eigh(A)
	return matmatmul(matmatmul(v, tf.matrix_diag(tf.log(u))), v, transpose_b=True)


def tensormatmul(S, x):
	if x.shape.ndims == 2:
		return tf.einsum('pqij,ij->pq')
	elif x.shape.ndims == 3:
		return tf.einsum('pqij,aij->apq')
	else:
		raise NotImplementedError

def mattensormul(x, S, transpose_x=False):
	idx_x = 'ij' if not transpose_x else 'ji'

	b_x = ['', 'a'][x.shape.ndims-2]

	if x.shape.ndims == 2:
		return tf.einsum('pqij,ij->pq')
	elif x.shape.ndims == 3:
		return tf.einsum('pqij,ij->apq')
	else:
		raise NotImplementedError

