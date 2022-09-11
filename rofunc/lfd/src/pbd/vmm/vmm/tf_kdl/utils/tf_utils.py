import tensorflow as tf

def matvecmul(mat, vec):
	"""
	Matrix-vector multiplication
	:param mat:
	:param vec:
	:return:
	"""
	if mat.shape.ndims == 2 and vec.shape.ndims == 1:
		return tf.einsum('ij,j->i', mat, vec)
	elif mat.shape.ndims == 2 and vec.shape.ndims == 2:
		return tf.einsum('ij,aj->ai', mat, vec)
	elif mat.shape.ndims == 3 and vec.shape.ndims == 1:
		return tf.einsum('aij,j->ai', mat, vec)
	elif mat.shape.ndims == 3 and vec.shape.ndims == 2:
		return tf.einsum('aij,aj->ai', mat, vec)
	else:
		raise NotImplementedError

def matmatmul(mat1, mat2):
	"""
	Matrix-matrix multiplication
	:param mat1:
	:param mat2:
	:return:
	"""
	if mat1.shape.ndims == 2 and mat2.shape.ndims == 2:
		return tf.matmul(mat1, mat2)
	elif mat1.shape.ndims == 3 and mat2.shape.ndims == 2:
		return tf.einsum('aij,jk->aik', mat1, mat2)
	elif mat1.shape.ndims == 2 and mat2.shape.ndims == 3:
		return tf.einsum('ij,ajk->aik', mat1, mat2)
	elif mat1.shape.ndims == 3 and mat2.shape.ndims == 3:
		return tf.einsum('aij,ajk->aik', mat1, mat2)
	else:
		raise NotImplementedError

def angular_vel_tensor(w):
	if w.shape.ndims == 1:
		return tf.stack([[0., -w[2], w[1]],
						 [w[2], 0. , -w[0]],
						 [-w[1], w[0], 0.]])
	else:
		di = tf.zeros_like(w[:, 0])
		return tf.transpose(
				tf.stack([[di, -w[:, 2], w[:, 1]],
						  [w[:, 2], di , -w[:, 0]],
						  [-w[:, 1], w[:, 0], di]]),
			perm=(2, 0, 1)
		)
