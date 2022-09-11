from .import_pykdl import *
import numpy as np

# @profile
def frame_to_np(frame, layout=None, vec=False):
	if vec:
		return np.array([
					frame.p[0], frame.p[1], frame.p[2],
					frame.M[0, 0], frame.M[1, 0], frame.M[2, 0],
					frame.M[0, 1], frame.M[1, 1], frame.M[2, 1],
					frame.M[0, 2], frame.M[1, 2], frame.M[2, 2],
				])
	else:
		return np.array([
			frame.p[0], frame.p[1], frame.p[2],
			frame.M[0, 0], frame.M[0, 1], frame.M[0, 2],
			frame.M[1, 0], frame.M[1, 1], frame.M[1, 2],
			frame.M[2, 0], frame.M[2, 1], frame.M[2, 2],
		])

def forward_kinematic(q, chain):
	nb_jnt = len(q) if isinstance(q, list) else q.shape[0]
	kdl_array = kdl.JntArray(nb_jnt)

	for j in range(nb_jnt):
		kdl_array[j] = q[j]

	end_frame = kdl.Frame()
	solver = kdl.ChainFkSolverPos_recursive(chain)

	solver.JntToCart(kdl_array, end_frame)

	return frame_to_np(end_frame)

class FKSolver(object):
	def __init__(self, chain, nb_jnt):
		"""

		:param chain:
		:param nb_jnt:	Number of joints
		:type nb_jnt:
		"""
		self.nb_jnt = nb_jnt
		self.chain = chain
		self.kdl_array = kdl.JntArray(nb_jnt)

		self.end_frame = kdl.Frame()
		self.solver = kdl.ChainFkSolverPos_recursive(chain)

	# @profile
	def solve(self, q, vec=False):
		for j in range(self.nb_jnt):
			self.kdl_array[j] = q[j]

		self.solver.JntToCart(self.kdl_array, self.end_frame)

		return frame_to_np(self.end_frame, vec=vec)


class JacSolver(object):
	def __init__(self, chain, nb_jnt):
		self.nb_jnt = nb_jnt
		self.chain = chain
		self.kdl_array = PyKDL.JntArray(nb_jnt)

		self.end_jacobian = PyKDL.Jacobian()
		self.solver = PyKDL.ChainJntToJacSolver(chain)

	# @profile
	def solve(self, q):
		for j in range(self.nb_jnt):
			self.kdl_array[j] = q[j]

		self.solver.JntToJac(self.kdl_array, self.end_frame)

		return frame_to_np(self.end_frame)


