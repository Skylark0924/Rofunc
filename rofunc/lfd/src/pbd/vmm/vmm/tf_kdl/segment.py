from .frame import Frame
from .joint import Joint
from .utils import *

class Segment(object):
	def __init__(self, joint, f_tip, child_name='', link=None):
		"""
		Segment of a kinematic chain

		:param joint:
		:type joint: tk.Joint
		:param f_tip:
		:type f_tip: tk.Frame
		"""
		self.joint = joint
		self.f_tip = joint.pose(0.).inv() *  f_tip
		self.child_name = child_name

		self.link = link

	def pose(self, q):
		return self.joint.pose(q) * self.f_tip

	def twist(self, q, qdot=0.):
		return self.joint.twist(qdot).ref_point(
			matvecmul(self.joint.pose(q).m, self.f_tip.p))

