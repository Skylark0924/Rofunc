import tensorflow as tf
from .joint import JointType
from .frame import Frame, Twist
from enum import IntEnum
import numpy as np
from .utils import FkLayout
from collections import OrderedDict
import matplotlib.pyplot as plt
from tensorflow_probability import distributions as ds
from .utils.plot_utils import axis_equal_3d

def stack_batch(vecs):
	"""
	Stack a list of vectors

	:param vecs: 	list of dim0 x tf.array((dim1, dim2)) or dim0 x tf.array((dim2,))
	:return: tf.array((dim1, dim0, dim2)) or tf.array(dim0, dim2)
	"""
	if vecs[0].shape.ndims < vecs[-1].shape.ndims:  # if first frame is not batch
		vecs[0] = tf.ones((tf.shape(vecs[-1])[0], 1)) * vecs[0]

	if vecs[-1].shape.ndims == 1:
		return tf.stack(vecs)
	elif vecs[-1].shape.ndims == 2:
		return tf.transpose(tf.stack(vecs), perm=(1, 0, 2))
	else:
		return NotImplementedError

def return_frame(p, layout=FkLayout.xm):
	"""
	Return a frame or list of frame in the desired layout

	:param p: 		tf_kdl.Frame or list of [tf_kdl.Frame]
	:param layout:	FkLayout
	:return:
	"""
	if layout is FkLayout.x:
		if isinstance(p, list):
			return stack_batch([_p.p for _p in p])
		else:
			return p.p
	elif layout is FkLayout.xq:
		if isinstance(p, list):
			return stack_batch([_p.xq for _p in p])
		else:
			return p.xq
	elif layout is FkLayout.xm:
		if isinstance(p, list):
			return stack_batch([_p.xm for _p in p])
		else:
			return p.xm
	elif layout is FkLayout.xmv:
		if isinstance(p, list):
			return stack_batch([_p.xmv for _p in p])
		else:
			return p.xmv
	elif layout is FkLayout.f:
		return p


class Chain(object):
	def	__init__(self, segments):
		"""
		Defines a kinematic Chain

		:param segments:
		"""
		self._segments = segments
		self.nb_segm = len(segments)
		self.nb_joint = len([seg for seg in segments if seg.joint.type != JointType.NoneT])

		self._joint_limits = None
		self._mean_pose = None
		self._masses = None
		self._mass = None

	@property
	def joint_limits(self):
		if self._joint_limits is None:
			self._joint_limits = [[seg.joint.limits['low'], seg.joint.limits['up']]
					for seg in self.segments if seg.joint.type != JointType.NoneT]

		return self._joint_limits

	def joint_limit_cost(self, q, std=0.1):
		return -ds.Normal(
			tf.constant(self.joint_limits, dtype=tf.float32)[:, 0], std).log_cdf(q) - ds.Normal(
			-tf.constant(self.joint_limits, dtype=tf.float32)[:, 1], std).log_cdf(-q)

	@property
	def mean_pose(self):
		if self._mean_pose is None:
			self._mean_pose= [np.mean([seg.joint.limits['low'], seg.joint.limits['up']])
								  for seg in self.segments if
								  seg.joint.type != JointType.NoneT]

		return self._mean_pose



	def constraints(self, q, dq=None, tau=None, alpha=1e5, s=0.1, relu=True,
					r_q=None, r_dq=None, r_tau=None):
		"""

		:param q: 		tf.Tensor((..., ndof))
		:param dq: 		tf.Tensor((..., ndof))
		:param tau: 	tf.Tensor((..., ndof))
		:param s: 		Distance at which the cost is < alpha * 1e-3
		:return:
		"""

		cost = []

		sp = lambda x: tf.nn.relu(x)

		if r_q is not None:
			q += tf.distributions.Normal(tf.zeros_like(q), r_q).sample()

		for _q, seg in zip(tf.unstack(q, axis=-1), self.segments):
			if seg.joint.type == JointType.NoneT:
				continue
			cost += [alpha * sp(_q - seg.joint.limits['up'])]
			cost += [alpha * sp(-_q + seg.joint.limits['low'])]

		if dq is not None:
			if r_dq is not None:
				dq += tf.distributions.Normal(tf.zeros_like(q), r_dq).sample()

			for _dq, seg in zip(tf.unstack(dq, axis=-1), self.segments):
				if seg.joint.type == JointType.NoneT:
					continue
				cost += [alpha * sp(tf.abs(_dq) - seg.joint.limits['vel'])]

		if tau is not None:
			if r_tau is not None:
				tau += tf.distributions.Normal(tf.zeros_like(q), r_tau).sample()

			for _tau, seg in zip(tf.unstack(dq, axis=-1), self.segments):
				if seg.joint.type == JointType.NoneT:
					continue
				cost += [alpha * sp(tf.abs(_tau) - seg.joint.limits['effort'])]

		return tf.reduce_sum(cost, axis=0)

	def plot(self, xs=None, qs=None, feed_dict=None, dim=None, ax=None, sess=None,
			 color='k', rng=None, view=None, proj=None, ax3d=None, remove_ax=False,
			 **kwargs):
		"""
		Give one or the other
			:param xs: kinematic chain as tensor (if not given will be computed
			:param qs: joint angles as numpy array

		:param feed_dict:
		:param rng : which range in the batch to plot
		:param view in ['front', 'top', 'side', '34']
		:param proj rotation matrix [3, 3]
		:return:
		"""
		sess = sess if sess is not None else tf.compat.v1.get_default_session()

		if isinstance(xs, np.ndarray):
			_xs = xs
		else:
			_xs = sess.run(xs, feed_dict=feed_dict)  # [batch_size, joint, dim]

		rng = range(0, _xs.shape[0]) if rng is None else rng

		label = kwargs.pop('label', None)

		if ax3d is not None:
			for _x in _xs[rng]:
				ax3d.plot(_x[:, 0], _x[:, 1], _x[:, 2],
						  marker='o', color=color, lw=10, mfc='w',
						  solid_capstyle='round', **kwargs)

			ax3d.plot([], [], [], marker='o', color=color, lw=10, mfc='w',
					  solid_capstyle='round',
					  **kwargs)

			if remove_ax:
				ax3d.set_xticks([])
				ax3d.set_yticks([])
				ax3d.set_zticks([])

			axis_equal_3d(ax3d)
			return

		if proj is None and view is None:

			if dim is None: dim = [0, 2]

			_xs_proj = _xs[:, :, dim]

		elif view is not None:
			assert view in ['front', 'top', 'side', '34'], "View should be in ['front', 'top', 'side', '34']"
			dim = {
				'front': [1, 2], 'top': [0, 1], 'side': [0, 2], '34' : [0, 0]
			}[view]
			if view != '34': _xs_proj = _xs[:, :, dim]
			else: _xs_proj = np.concatenate(
				[0.71 * _xs[:, :, [0]] + 0.71 * _xs[:, :, [1]], _xs[:, :, [2]]], axis=2)

		else:
			raise NotImplementedError("Projection matrix is not implemented yet, please do it and commit !")


		for x in _xs_proj[rng]:
			plot = ax if ax is not None else plt
			plot.plot(x[:, 0], x[:, 1], marker='o', color=color, lw=10, mfc='w',
					 solid_capstyle='round',
					 **kwargs)

		kwargs['alpha'] = 1.; kwargs['label'] = label
		plot.plot([], [], marker='o', color=color, lw=10, mfc='w',
				 solid_capstyle='round',
				 **kwargs)

		if ax is not None: ax.set_aspect('equal')
		else: plt.axes().set_aspect('equal')

	@property
	def segments(self):
		return self._segments

	def ee_frame(self, q, n=0, layout=FkLayout.xm):
		"""
		Pose of last-n segment of the cain

		:param q:		tf.Tensor()
			Joint angles
		:param n:		int
			index from end of segment to get
		:param layout:
			layout of frame
		:return:
		"""

		if q.shape.ndims == 1:
			p = self.segments[0].pose(q[0])
		elif q.shape.ndims == 2:
			p = self.segments[0].pose(q[:, 0])

		j = 1

		for i in range(1, self.nb_segm-n):
			if self.segments[i].joint.type is not JointType.NoneT:
				if q.shape.ndims == 1:
					p = p * self.segments[i].pose(q[j])
				elif q.shape.ndims == 2:
					p = p * self.segments[i].pose(q[:, j])
				else:
					raise NotImplementedError
				j += 1
			else:
					p = p * self.segments[i].pose(0.)


		return return_frame(p, layout)

	@property
	def masses(self):
		if self._masses is None:
			self._masses = tf.constant(
				[seg.link.mass for seg in self.segments if seg.link is not None])

		return self._masses

	@property
	def mass(self):
		if self._mass is None:
			self._mass = sum(
				[seg.link.mass for seg in self.segments if seg.link is not None]
			)
		return self._mass

	def xs(self, q, layout=FkLayout.xm, floating_base=None, get_links=False,
		   get_collision_samples=False, sample_size=10):
		"""
		Pose of all segments of the chain

		:param q:		[batch_size, nb_joint] or [nb_joint] or list of [batch_size]
			Joint angles
		:param layout:
			layout of frame
		:param floating_base Frame() or tuple (p translation vector, m rotation matrix)
		:param get_links: get forward kinematics of links and center of mass
		:return:
		"""
		if floating_base is None:
			p = [Frame()]
		elif isinstance(floating_base, tuple) or isinstance(floating_base, list):
			p = [Frame(p=floating_base[0], m=floating_base[1])]
		elif isinstance(floating_base, Frame):
			p = [floating_base]
		else:
			raise ValueError("Unknown floating base type")

		j = 0
		links = [p[0]]

		for i in range(self.nb_segm):
			if self.segments[i].joint.type is not JointType.NoneT:
				if isinstance(q, list) or q.shape.ndims == 1:
					p += [p[-1] * self.segments[i].pose(q[j])]
					if (get_links or get_collision_samples) and self.segments[i].link is not None:
						links += [p[-1] * self.segments[i].link.frame]
				elif q.shape.ndims == 2:
					p += [p[-1] * self.segments[i].pose(q[:, j])]
					if (get_links or get_collision_samples) and self.segments[i].link is not None:
						links += [p[-1] * self.segments[i].link.frame]
				else:
					raise NotImplementedError
				j += 1
			else:
				p += [p[-1] * self.segments[i].pose(0.)]
				if (get_links or get_collision_samples) and self.segments[i].link is not None:
					links += [p[-1] * self.segments[i].link.frame]

		if get_links or get_collision_samples:
			# TODO check for mass of first segment


			# compute center of mass

			links_stacked = return_frame(links, layout)

			com  = tf.reduce_sum(
				self.masses[None, :, None] * links_stacked[..., 1:, :3], axis=-2)/self.mass

			if get_collision_samples:
				samples = []
				i = 1
				for seg in self.segments:
					if seg.link	is not None and seg.link.collision_mesh is not None:
						_samples = seg.link.collision_mesh.sample(sample_size)
						samples += [links[i] * _samples]

					i += 1

				if samples[0].shape.ndims == 2 and samples[1].shape.ndims == 3: # make batch_dimension of first joint
					samples[0] = samples[0][None] * tf.ones_like(samples[1][:, :, 0:1])


				if get_links:
					return return_frame(p, layout), links_stacked, com, tf.stack(samples)
				else:
					return return_frame(p, layout), tf.stack(samples)
			else:
				return return_frame(p, layout), links_stacked, com
		else:
			return return_frame(p, layout)

	def jacobian(self, q, n=0, layout=FkLayout.xm, floating_base=None):
		"""

		:param q:
		:param n: 	segment to take counting from the end. E.g. 0 is for the last
		:param mat:
		:param vec: use a flattening of rotation matrix as [x_1, x_2, x_3, y_1, ..., z_3]
		:return:
		"""
		# TODO is very slow !! try another implementation

		# init base frame
		is_batch = not isinstance(q, list) and q.shape.ndims == 2

		if floating_base is None:
			T_tmp = Frame()
		elif isinstance(floating_base, tuple) or isinstance(floating_base, list):
			T_tmp = Frame(p=floating_base[0], m=floating_base[1])
		elif isinstance(floating_base, Frame):
			T_tmp = floating_base
		else:
			raise ValueError("Unknown floating base type")


		# create zero twist for each joint
		# jac = [Twist() for i in range(self.nb_segm-n)]

		j = 0
		jac = []
		for i in range(self.nb_segm-n):
			# get pose of following segment
			if self.segments[i].joint.type is not JointType.NoneT:
				_q = q[:, j] if is_batch else q[j]
				total = T_tmp  * self.segments[i].pose(_q)
				t_tmp = T_tmp.m * self.segments[i].twist(_q, 1.0)
			else:
				total = T_tmp  * self.segments[i].pose(0.)

			# get rotated twist of following segment
			# Changing reference point of all columns to new EE
			# jac = [_t.ref_point(total.p - T_tmp.p)
			# 		  for _t in jac[:i]] + [jac[i:]]
			if len(jac):
				jac = [_t.ref_point(total.p - T_tmp.p)
						  for _t in jac]

			# jac[i] = t_tmp

			if self.segments[i].joint.type is not JointType.NoneT:
				j += 1
				jac += [t_tmp]

			T_tmp = total

		if layout in [FkLayout.xm, FkLayout.xmv]:
			return tf.transpose(stack_batch([_t.dx_mat(total.m, layout=layout) for _t in jac]), (0, 2, 1))
		elif layout in [FkLayout.xq]:
			return tf.transpose(stack_batch([_t.dx for _t in jac]), (0, 2, 1))
		elif layout in [FkLayout.x]:
			return tf.transpose(stack_batch([_t.dx for _t in jac]), (0, 2, 1))[:, :3]
		else:
			raise NotImplementedError

class ChainDict(OrderedDict, Chain):
	_names = None
	_unique_names = None
	_nb_joints = None
	_joint_limits = None
	_mass = None

	@property
	def actuated_joint_names(self):
		if self._names is None or self._unique_names is None:
			self._names = []
			self._unique_names = []
			for chain in self.values():
				self._names += [seg.child_name for seg in chain._segments if
						  seg.joint.type != JointType.NoneT]
				self._unique_names += [seg.child_name for seg in chain._segments if
						  seg.joint.type != JointType.NoneT and not seg.child_name in self._unique_names]
			# self._unique_names = list(set(self._names))

			self._nb_joints = len(self._unique_names)

		return self._names

	@property
	def nb_joint(self):
		if self._nb_joints is None:
			self.actuated_joint_names

		return self._nb_joints

	@property
	def joint_limits(self):
		if self._joint_limits is None:
			self.actuated_joint_names

			_joint_limits_all = {}
			for name, chain in self.iteritems():
				for seg in chain.segments:
					if seg.joint.type == JointType.NoneT: continue
					_joint_limits_all[seg.child_name] = [seg.joint.limits['low'], seg.joint.limits['up']]


			self._joint_limits = [_joint_limits_all[name] for name in self._unique_names]

		return self._joint_limits



	@property
	def mass(self):
		if self._mass is None:
			self._mass = sum([chain.mass for name, chain in self.iteritems()])

		return self._mass

	def plot(self, xs=None, qs=None, *args, **kwargs):
		if xs is None: raise NotImplementedError

		sess = kwargs.get('sess', tf.get_default_session())
		feed_dict = kwargs.get('feed_dict', {})

		# evaluating all the chains here in common is because of random seed that is
		# different for each chains and results in disconnected robot

		if isinstance(xs.values()[0], np.ndarray):
			_xs = xs
		else:
			_xs = {name: val for name, val in
		 		zip(self, sess.run([xs[name] for name in self], feed_dict=feed_dict))}

		for name, chain in self.iteritems():
			chain.plot(xs=_xs[name], *args, **kwargs)
			kwargs.pop('label', None)

	def jacobian(self, q, n=0, layout=FkLayout.xm, floating_base=None, name=None):
		self.actuated_joint_names

		idx_chain = {
			name: [
				self._unique_names.index(seg.child_name)
				for seg in chain._segments if seg.joint.type != JointType.NoneT
			] for name, chain in self.iteritems()
		}

		q_chains = {
			name: [q[:, i] for i in idx]
			for name, idx in idx_chain.iteritems()
		}

		if name is None:
			xs_chains = {
				name : chain.jacobian(
					q_chains[name], layout=layout,  n=n, floating_base=floating_base)
				for name, chain in self.iteritems()
			}

			xs_chains_aug = {}

			for name, jac in xs_chains.iteritems():

				jacs = [
					tf.zeros_like(xs_chains[name][:, :, 0])[:, :, None]
					for i in range(self.nb_joint)
				]

				for j, i in enumerate(idx_chain[name]):
					jacs[i] = xs_chains[name][:, :, j][:, :, None]

				xs_chains_aug[name] = tf.concat(
					jacs, axis=-1
				)

			return xs_chains_aug
		else:
			return self[name].jacobian(
				q_chains[name], layout=layout, n=n, floating_base=floating_base)


	def xs(self, q, layout=FkLayout.xm, floating_base=None, name=None, get_links=False):
		"""

		:param q: [batch_shape, self.nb_joint] or [self.nb_joint]
		:param layout:
		:param name:  If you want to retrieve only a chain
		:return:
		"""
		self.actuated_joint_names

		idx_chain = {
			name: [
				self._unique_names.index(seg.child_name)
				for seg in chain._segments if seg.joint.type != JointType.NoneT
			] for name, chain in self.iteritems()
			   }

		q_chains = {
			name: [q[:, i] for i in idx]
			for name, idx in idx_chain.iteritems()
		}

		if name is None:
			xs_chains = {
				name : chain.xs(
					q_chains[name], layout=layout, floating_base=floating_base, get_links=get_links)
				for name, chain in self.iteritems()
			}

			if not get_links:
				return xs_chains
			else:
				# compute total center of mass
				mass = tf.reduce_sum(
					[xs_chains[name][2] * self[name].mass for name, chain in self.iteritems()],
					axis=0)/self.mass


				return {name: xs_chains[name][0] for name, chain in self.iteritems()},\
					   {name: xs_chains[name][1] for name, chain in self.iteritems()},\
						mass


		else:
			return self[name].xs(
				q_chains[name], layout=layout, floating_base=floating_base,
				get_links=get_links)