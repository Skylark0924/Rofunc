from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import factorial

plt.style.use('ggplot')

import scipy.sparse as ss


def quaternion_from_matrix(matrix, isprecise=False):
	"""Return quaternion from rotation matrix.

	If isprecise is True, the input matrix is assumed to be a precise rotation
	matrix and a faster algorithm is used.
	"""
	M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
	if isprecise:
		q = np.empty((4,))
		t = np.trace(M)
		if t > M[3, 3]:
			q[0] = t
			q[3] = M[1, 0] - M[0, 1]
			q[2] = M[0, 2] - M[2, 0]
			q[1] = M[2, 1] - M[1, 2]
		else:
			i, j, k = 1, 2, 3
			if M[1, 1] > M[0, 0]:
				i, j, k = 2, 3, 1
			if M[2, 2] > M[i, i]:
				i, j, k = 3, 1, 2
			t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
			q[i] = t
			q[j] = M[i, j] + M[j, i]
			q[k] = M[k, i] + M[i, k]
			q[3] = M[k, j] - M[j, k]
		q *= 0.5 / np.sqrt(t * M[3, 3])
	else:
		m00 = M[0, 0]
		m01 = M[0, 1]
		m02 = M[0, 2]
		m10 = M[1, 0]
		m11 = M[1, 1]
		m12 = M[1, 2]
		m20 = M[2, 0]
		m21 = M[2, 1]
		m22 = M[2, 2]
		# symmetric matrix K
		K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
					  [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
					  [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
					  [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
		K /= 3.0
		# quaternion is eigenvector of K that corresponds to largest eigenvalue
		w, V = np.linalg.eigh(K)
		q = V[:, np.argmax(w)]
	if q[0] < 0.0:
		np.negative(q, q)
	return q


def get_canonical(nb_dim, nb_deriv=2, dt=0.01):
	A1d = np.zeros((nb_deriv, nb_deriv))

	for i in range(nb_deriv):
		A1d += np.diag(np.ones(nb_deriv - i), i) * np.power(dt, i) / factorial(i)

	B1d = np.zeros((nb_deriv, 1))
	for i in range(1, nb_deriv + 1):
		B1d[nb_deriv - i] = np.power(dt, i) / factorial(i)

	return np.kron(A1d, np.eye(nb_dim)), np.kron(B1d, np.eye(nb_dim))

def multi_timestep_matrix(A, B, nb_step=4):
	xi_dim, u_dim = A.shape[0], B.shape[1]
	_A = np.zeros((xi_dim * nb_step, xi_dim * nb_step))
	_B = np.zeros((xi_dim * nb_step, u_dim))

	_A[:xi_dim, :xi_dim] = A

	for i in range(1, nb_step):
		_A[xi_dim * i:xi_dim * (i + 1), xi_dim * (i - 1):xi_dim * i] = np.eye(xi_dim)

	_B[:xi_dim, :u_dim] = B
	return _A, _B


def fd_transform(d, xi_dim, nb_past, dt=0.1):
	"""
	Finite difference transform matrix

	:param d:
	:param xi_dim:
	:param nb_past:
	:param dt:
	:return:
	"""

	T_1 = np.zeros((xi_dim * nb_past, xi_dim * (nb_past - d)))

	for i in range(nb_past - d):
		T_1[xi_dim * i:xi_dim * (i + 1), xi_dim * (i):xi_dim * (i + 1)] = np.eye(
			xi_dim) * dt ** d

		nb = [[1],
			  [1, -1],
			  [1., -2, 1],
			  [1., -3, 3, -1],
			  [1., -4., 6., -4., 1.]]

		for j in range(d):
			T_1[xi_dim * (i + 1 + j):xi_dim * (i + 2 + j), xi_dim * i:xi_dim * (i + 1)] = \
				nb[d][j + 1] * np.eye(xi_dim) * dt ** d

	return T_1

def multi_timestep_fd_q(rs, xi_dim, dt):
	"""

	:param rs: list of std deviations of derivatives
	:param xi_dim:
	:param nb_past:
	:param dt:
	:return:
	"""
	nb_past = len(rs)

	Qs = []
	for i in range(nb_past):
		T = fd_transform(i + 1, xi_dim, nb_past, dt)
		Q = np.eye((xi_dim * (nb_past - i - 1))) * rs[i] ** -2
		Qs += [T.dot(Q).dot(T.T)]

	return np.sum(Qs, axis=0)


def lifted_noise_matrix(A=None, B=None, nb_dim=3, dt=0.01, horizon=50):
	r"""
	Given a linear system with white noise, as in LQG,

	.. math::
		\xi_{t+1} = \mathbf{A} (\xi_t + w_i) + \mathbf{B} u_t + v_i

	returns the lifted form for noise addition, s_v, s_w,

	.. math::
	    \mathbf{\xi} = \mathbf{S}_{\xi} \xi_0 + \mathbf{S}_u \mathbf{u}
	    + \mathbf{S}_v + \mathbf{S}_w

	:return: s_u
	"""
	if A is None or B is None:
		A, B = get_canonical(nb_dim, 2, dt)

	s_v = np.zeros((A.shape[0] * horizon, A.shape[0] * horizon))

	A_p = np.eye(A.shape[0])
	At_b_tmp = []
	for i in range(horizon):
		# s_xi[i * A.shape[0]:(i + 1) * A.shape[0]] = A_p
		At_b_tmp += [A_p]
		A_p = A_p.dot(A)

	for i in range(horizon):
		for j in range(i + 1):
			s_v[i * A.shape[0]:(i + 1) * A.shape[0], j * A.shape[1]:(j + 1) * A.shape[1]] = \
			At_b_tmp[i - j - 1]

	return s_v


def lifted_transfer_matrix(A=None, B=None, nb_dim=3, dt=0.01, horizon=50, sparse=False):
	r"""
	Given a linear system

	.. math::
		\xi_{t+1} = \mathbf{A} \xi_t + \mathbf{B} u_t

	returns the lifted form for T timesteps

	.. math::
	    \mathbf{\xi} = \mathbf{S}_{\xi} \xi_0 + \mathbf{S}_u \mathbf{u}


	"""

	if A is None or B is None:
		A, B = get_canonical(nb_dim, 2, dt)

	s_xi = np.zeros((A.shape[0] * horizon, A.shape[1]))
	A_p = np.eye(A.shape[0])
	At_b_tmp = []
	for i in range(horizon):
		s_xi[i * A.shape[0]:(i + 1) * A.shape[0]] = A_p
		At_b_tmp += [np.copy(A_p.dot(B))]
		A_p = A_p.dot(A)

	s_u = np.zeros((B.shape[0] * horizon, B.shape[1] * horizon))

	for i in range(horizon):
		for j in range(i):
			s_u[i * B.shape[0]:(i + 1) * B.shape[0], j * B.shape[1]:(j + 1) * B.shape[1]] = \
			At_b_tmp[i - j - 1]

	if sparse:
		return ss.csc_matrix(s_xi), ss.csc_matrix(s_u)
	else:
		return s_xi, s_u


def gu_pinv(A, rcond=1e-15):
	I = A.shape[0]
	J = A.shape[1]
	return np.array([[np.linalg.pinv(A[i, j]) for j in range(J)] for i in range(I)])


def create_relative_time(q, start=-1.):
	"""
	:param 	q:		[list of int]
		List of state indicator.
		ex: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, ...]
	:return time:	[np.array(nb_timestep,)]
		Phase for each of the timestep
	"""
	# find the index of states changes
	state_idx = np.array([-1] + np.nonzero(np.diff(q))[0].tolist() + [len(q) - 1])

	time = np.zeros(len(q))

	for i, t in enumerate(state_idx[:-1]):
		start_phase = start if i == 0 else -1.
		l = state_idx[i + 1] - state_idx[i]
		time[state_idx[i] + 1:state_idx[i + 1] + 1] = np.linspace(start_phase, 1, l)

	return time, state_idx


def align_trajectories_hsmm(data, nb_states=5):
	from ..hsmm import HSMM

	if data[0].ndim > 2:  # if more than rank 2, flatten last dims
		data_vectorized = [np.reshape(d, (d.shape[0], -1)) for d in data]
	else:
		data_vectorized = data

	model = HSMM(nb_dim=data[0].shape[1], nb_states=nb_states)
	model.init_hmm_kbins(data_vectorized)


	qs = [model.viterbi(d) for d in data_vectorized]

	time, sqs = zip(*[create_relative_time(q) for q in qs])

	start_idx = [np.array((np.nonzero(np.diff(q))[0] + 1).tolist()) for q in qs]

	for s_idxs, t in zip(start_idx, time):
		for s_idx in s_idxs:
			t[s_idx:] += 2. + (t[s_idx + 1] - t[s_idx])

	return time


def align_trajectories(data, additional_data=[], hsmm=True, nb_states=5):
	"""

	:param data: 		[list of np.array([nb_timestep, M, N, ...])]
	:return:
	"""
	from dtw import dtw
	if hsmm:
		time = align_trajectories_hsmm(data, nb_states)

	ls = np.argmax([d.shape[0] for d in data])  # select longest as basis

	data_warp = []
	additional_data_warp = [[] for d in additional_data]

	for j, d in enumerate(data):
		if hsmm:
			dist, cost, acc, path = dtw(time[ls], time[j],
										dist=lambda x, y: np.linalg.norm(x - y))
		else:
			dist, cost, acc, path = dtw(data[ls], d,
										dist=lambda x, y: np.linalg.norm(x - y, ord=1))

		data_warp += [d[path[1]][:data[ls].shape[0]]]

		for i, ad in enumerate(additional_data):
			additional_data_warp[i] += [ad[j][path[1]][:data[ls].shape[0]]]

	if len(additional_data):
		return [data_warp] + additional_data_warp
	else:
		return data_warp


def angle_to_rotation(theta):
	return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def feature_to_slice(nb_dim=2, nb_frames=None, nb_attractor=2,
					 features=None):
	# type: (int, list of int, int, list of list of string) -> object
	index = []
	l = 0
	for i, nb_frame, feature in zip(range(nb_attractor), nb_frames, features):
		index += [[]]
		for m in range(nb_frame):
			index[i] += [{}]
			for f in feature:
				index[i][m][f] = slice(l, l + nb_dim)
				l += nb_dim

	return index


def dtype_to_index(dtype):
	last_idx = 0
	idx = {}
	for name in dtype.names:
		idx[name] = range(last_idx, last_idx + dtype[name].shape[0])
		last_idx += dtype[name].shape[0]

	return idx


def gu_pinv(A, rcond=1e-15):
	I = A.shape[0]
	J = A.shape[1]
	return np.array([[np.linalg.pinv(A[i, j]) for j in range(J)] for i in range(I)])


#
# def gu_pinv(a, rcond=1e-15):
#     a = np.asarray(a)
#     swap = np.arange(a.ndim)
#     swap[[-2, -1]] = swap[[-1, -2]]
#     u, s, v = np.linalg.svd(a)
#     cutoff = np.maximum.reduce(s, axis=-1, keepdims=True) * rcond
#     mask = s > cutoff
#     s[mask] = 1. / s[mask]
#     s[~mask] = 0
#
#     return np.einsum('...uv,...vw->...uw',
#                      np.transpose(v, swap) * s[..., None, :],
#                      np.transpose(u, swap))

def plot_model_time(model, demos, figsize=(10, 2), dim_idx=[1], demo_idx=0):
	nb_dim = len(dim_idx)
	nb_samples = len(demos)

	fig = plt.figure(3, figsize=(figsize[0], figsize[1] * nb_dim))
	# fig.suptitle("Reproduction", fontsize=14, fontweight='bold')
	nb_plt = nb_dim
	ax = []  # subplots
	label_size = 15
	### specify subplots ###
	gs = gridspec.GridSpec(nb_dim, 1)

	for j in range(nb_plt):  # [0, 2, 1, 3, ...]
		ax.append(fig.add_subplot(gs[j]))
	for a in ax:
		a.set_axis_bgcolor('white')

	fig.suptitle("Demonstration", fontsize=14, fontweight='bold')

	idx = np.floor(np.linspace(1, 255, model.nb_states)).astype(int)
	color = cmap.viridis(range(256))[idx, 0:3]  # for states

	state_sequ = []
	for d in demos:
		state_sequ += [model.viterbi(d['Data'])]

	d = demos[demo_idx]
	s = state_sequ[demo_idx]

	for dim, a in zip(dim_idx, ax):
		a.plot(d['Data'][dim, :])

		for x_s, x_e, state in zip([0] + np.where(np.diff(s))[0].tolist(),  # start step
								   np.where(np.diff(s))[0].tolist() + [len(s)],  # end step
								   np.array(s)[[0] + (
									   np.where(np.diff(s))[0] + 1).tolist()]):  # state idx
			a.axvline(x=x_e, ymin=0, ymax=1, c='k', lw=2, ls='--')

			mean = model.Mu[dim, state]
			var = np.sqrt(model.Sigma[dim, dim, state])
			a.plot([x_s, x_e], [mean, mean], c='k', lw=2)

			a.fill_between([x_s, x_e], [mean + var, mean + var], [mean - var, mean - var],
						   alpha=0.5, color=color[state])

	plt.show()


def quaternion_matrix(quaternion):
	"""Return homogeneous rotation matrix from quaternion.

	>>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
	>>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
	True
	>>> M = quaternion_matrix([1, 0, 0, 0])
	>>> np.allclose(M, np.identity(4))
	True
	>>> M = quaternion_matrix([0, 1, 0, 0])
	>>> np.allclose(M, np.diag([1, -1, -1, 1]))
	True

	"""
	q = np.array(quaternion, dtype=np.float64, copy=True)
	q = q[[3, 0, 1, 2]]

	n = np.dot(q, q)
	if n < np.finfo(float).eps * 4.0:
		return np.identity(4)
	q *= np.sqrt(2.0 / n)
	q = np.outer(q, q)
	return np.array([
		[1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
		[q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
		[q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
		[0.0, 0.0, 0.0, 1.0]])


def plot_demos_3d(demos, figsize=(15, 5), angle=[60, 45]):
	nb_samples = len(demos)
	fig = plt.figure(1, figsize=figsize)
	fig.suptitle("Demonstration", fontsize=14, fontweight='bold')
	nb_plt = 2
	ax = []
	label_size = 15

	idx = np.floor(np.linspace(1, 255, nb_samples)).astype(int)
	color_demo = cmap.viridis(range(256))[idx, 0:3]  # for states

	nb = 0
	gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

	for j in [0, 1]:
		ax.append(fig.add_subplot(gs[j], projection='3d', axisbg='white'))

	ax[nb].set_title(r'$\mathrm{Skill\ A}$')
	for ax_ in ax:
		ax_.view_init(angle[0], angle[1])

	for i, c in zip(range(nb_samples), color_demo):
		a = 1
		ax[nb].plot(demos[i]['Data'][0, :], demos[i]['Data'][1, :], demos[i]['Data'][2, :],
					color=c, lw=1, alpha=a)
	# ax[nb].plot(demos[i]['Data'][7,:], demos[i]['Data'][8,:],'H',color=c,ms=10,alpha=a)

	nb += 1
	ax[nb].set_title(r'$\mathrm{Skill\ B}$')
	for i, c in zip(range(nb_samples), color_demo):
		a = 1
		ax[nb].plot(demos[i]['Data'][3, :], demos[i]['Data'][4, :], demos[i]['Data'][5, :],
					color=c, lw=1, alpha=a)


# ax[nb].plot(demos[i]['Data'][7,:], demos[i]['Data'][8,:],'H',color=c,ms=10,alpha=a)


def repro_plot(model, demos, save=False, tp_list=[], figsize=(3.5, 5)):
	nb_states = model.nb_states
	nb_tp = len(tp_list)
	idx = np.floor(np.linspace(1, 255, model.nb_states)).astype(int)
	color = cmap.viridis(range(256))[idx, 0:3]  # for states

	fig = plt.figure(3, figsize=(figsize[0] * nb_tp, figsize[1]))
	# fig.suptitle("Reproduction", fontsize=14, fontweight='bold')
	nb_plt = nb_tp * 2
	ax = []  # subplots
	label_size = 15
	t = 50  # timestep for reproduction

	# regress in first configuration
	i_in = [6, 7, 8]  # input dimension
	i_out = [0, 1, 2]  # output

	### specify subplots ###
	gs = gridspec.GridSpec(2, nb_tp, height_ratios=[4, 1])

	rn = []
	for i in range(nb_tp):
		rn += [i, i + nb_tp]

	for j in rn:  # [0, 2, 1, 3, ...]
		ax.append(fig.add_subplot(gs[j]))
	for a in ax:
		a.set_axis_bgcolor('white')

	for i in range(0, nb_tp * 2, 2):
		tp = tp_list[i / 2]
		data_in = tp[0]['b']
		model.regress(data_in - tp[1]['b'], i_in, i_out)
		prod_1 = model.prodgmm(tp)

		nb = i  # subplots counter
		ax[nb].set_title(r'$\mathrm{(a)}$')

		item_plt, = ax[nb].plot(data_in[0], data_in[1], '^', color=color[3], ms=12)

		pblt.plot_gmm(prod_1.Mu, prod_1.Sigma, dim=[0, 1], color=color,
					  alpha=model.PriorsR * nb_states, ax=ax[nb], nb=2)

		### plot state sequence ###
		nb = i + 1

		### get state sequence ###
		h = model.forward_variable_priors(t, model.PriorsR, start_priors=model.StatesPriors)

		for i in range(nb_states):
			ax[nb].plot(h[i, :], color=color[i])

	"""LEGEND, LABEL, ..."""
	for i in range(0, nb_plt, 2):
		# rob_plt, = ax[i].plot(40,40,'s',color=(1,0.4,0),ms=8,zorder=30)
		ax[i].set_aspect('equal', 'datalim')
		for j in [3, 4, 5, 6, 2]:
			demo_plt, = ax[i].plot(demos[j]['Glb'][0, :], demos[j]['Glb'][1, :], 'k:', lw=1,
								   alpha=1)

	for i in range(1, nb_plt, 2):
		#     ax[i].set_title(r'$\mathrm{forward\ variable}\, \alpha_t(z_n)$')
		ax[i].set_title(r'$\alpha_t(z_n)$', fontsize=16)
		ax[i].set_xlabel(r'$t\, \mathrm{[timestep]}$', fontsize=16)
		ax[i].set_ylim([-0.1, 1.1])
		ax[i].set_yticks(np.linspace(0, 1, 3))

	lgd = fig.legend([item_plt, demo_plt], ['obstacle position', 'Demonstrations']
					 , frameon=True, ncol=3,
					 bbox_to_anchor=(0.1, -0.01), loc='lower left', numpoints=1)
	frame = lgd.get_frame()
	# frame.set_facecolor('White')


	plt.tight_layout(pad=2.4, w_pad=0.9, h_pad=1.0)
	if save:
		plt.savefig('/home/idiap/epignat/thesis/paper/images/' + skill_name + '_repro.pdf',
					bbox_extra_artists=(lgd,), bbox_inches='tight')
	plt.show()


def plot_model(model, demos, figsize=(8, 3.5), skill_name='temp', save=False):
	nb_samples = len(demos)
	fig = plt.figure(2, figsize=figsize)
	# fig.suptitle("Model", fontsize=14, fontweight='bold')
	nb_plt = 3
	ax = []
	label_size = 15
	# plt.style.use('bmh')
	plt.style.use('ggplot')

	idx = np.floor(np.linspace(1, 255, model.nb_states)).astype(int)
	color = cmap.viridis(range(256))[idx, 0:3]  # for states

	nb = 0

	gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.8])

	for j in range(nb_plt):
		ax.append(fig.add_subplot(gs[j]))
		ax[j].set_axis_bgcolor('white')

	ax[nb].set_title(r'$(a)\ j=1$')

	# for i in range(nb_samples):
	#    ax[nb].plot(demos[i]['Data'][4,0], demos[i]['Data'][5,0],'^',color=color[3],ms=10,alpha=0.5,zorder=30)


	for i in range(nb_samples):
		ax[nb].plot(demos[i]['Data'][0, :], demos[i]['Data'][1, :], 'k:', lw=1, alpha=1)

	pblt.plot_gmm(model.Mu, model.Sigma, dim=[0, 1], color=color, alpha=0.8, linewidth=1,
				  ax=ax[nb], nb=1)

	ax[nb].set_ylabel('z position [cm]')
	nb += 1

	ax[nb].set_title(r'$(b)\ j=2$')

	for i in range(nb_samples):
		demos_plt, = ax[nb].plot(demos[i]['Data'][3, :], demos[i]['Data'][4, :], 'k:', lw=1,
								 alpha=1)
	# ax[nb].plot(demos[i]['Data'][2,0], demos[i]['Data'][3,0],'H',color=c,ms=10)

	pblt.plot_gmm(model.Mu, model.Sigma, dim=[3, 4], color=color, alpha=0.8, linewidth=1,
				  ax=ax[nb], nb=1)

	nb += 1
	ax[nb].set_title(r'$(c)\ \mathrm{sensory}$')

	for i in range(nb_samples):
		sense_plt, = ax[nb].plot(demos[i]['Data'][6, 0], demos[i]['Data'][7, 0], '^',
								 color=color[3], ms=12, zorder=30)

	pblt.plot_gmm(model.Mu, model.Sigma, dim=[6, 7], color=color, alpha=0.5, ax=ax[nb],
				  nb=1)
	# ax[nb].set_xlim([-20,140])

	plt.tight_layout()

	lgd = fig.legend([demos_plt, sense_plt],
					 ['demonstrations', 'hand position'], frameon=True, ncol=2,
					 bbox_to_anchor=(0.4, -0.01), loc='lower left', numpoints=1)
	# frame = lgd.get_frame()
	# frame.set_facecolor('White')

	for i in range(nb_plt):
		# ax[i].plot(0, 0,'+',color='k',ms=20,zorder=30,lw=2)

		ax[i].set_xlabel('x position [cm]')

	plt.tight_layout(pad=2.8, w_pad=0.2, h_pad=-1.0)
	if save:
		plt.savefig('/home/idiap/epignat/thesis/paper/images/' + skill_name + '_model.pdf',
					bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_demos(demos, data_dim, figsize=(8, 5)):
	nb_samples = len(demos)
	fig = plt.figure(2, figsize=figsize)
	# fig.suptitle("Model", fontsize=14, fontweight='bold')
	nb_plt = len(data_dim)
	ax = []
	label_size = 15
	# plt.style.use('bmh')
	plt.style.use('ggplot')
	nb = 0

	gs = gridspec.GridSpec(nb_plt, 1)

	for j in range(nb_plt):
		ax.append(fig.add_subplot(gs[j]))
		ax[j].set_axis_bgcolor('white')

	for j, dim in enumerate(data_dim):
		for i in range(nb_samples):
			ax[j].plot(demos[i]['Data'][dim, :].T)


def train_test(demos, demo_idx=0, nb_states=5, test=True, sensory=True, kbins=True,
			   hmmr=True,
			   nb_dim=3, nb_frames=2):
	demos_train = deepcopy(demos)
	nb_samples = len(demos)
	if test:
		demos_train.pop(demo_idx)
		nb_s = nb_samples - 1
	else:
		nb_s = nb_samples

	model = pbd.TP_HMM(nb_states, nb_dim=nb_dim, nb_frames=nb_frames)
	dep = [[0, 1], [2, 3], [4, 5]]

	Data_train = np.hstack([d['Data'] for d in demos_train])
	# model.init_hmm_kmeans(Data, nb_states, nb_samples, dep=dep)

	best = {'model': None, 'score': np.inf}
	for i in range(10):
		if sensory:
			model.init_hmm_gmm(Data_train, nb_states, nb_samples, dep=dep)
			scale = 8.
		else:
			if kbins:
				model.init_hmm_kbins(Data_train, nb_states, nb_s, dep=dep)
			else:
				model.init_hmm_kmeans(Data_train, nb_states, nb_samples, dep=dep,
									  dim_init=range(6))
			scale = 1e10

		if sensory:
			score = model.em_hmm(demos_train, dep=dep, reg=0.0002,
								 reg_diag=[1., 1., 1., 1., 1., 1., scale, scale, scale])
		else:
			score = model.em_hmm(demos_train, dep=dep, reg=0.0002,
								 reg_diag=[1., 1., 1., 1., 1., 1., scale, scale, scale],
								 end_cov=True)
		if score < best['score']:
			best['score'] = score

			best['model'] = deepcopy(model)

	print('Best :', best['score'])
	model = best['model']

	model.compute_duration(demos_train)

	# model.init_hmm_kbins(Data, nb_states, nb_samples, dep=dep)
	if hmmr:
		hmmr = pbd.hmmr.HMMR(model, nb_dim=3)

		min_dist = pow(5e-2, 3)
		hmmr.to_gmr(demos_train, mix_std=0.1, reg=min_dist, plot_on=False)
	else:
		hmmr = None

	return model, hmmr


def repro_demo(model, hmmr, demos, demo_idx=0, start_point=None, plot_on=False):
	nb_states = model.nb_states
	nb_samples = len(demos)

	t = 50  # timestep for reproduction
	# regress in first configuration
	i_in = [6, 7, 8]  # input dimension
	i_out = [0, 1, 2]  # output

	tp = deepcopy(demos[demo_idx]['TPs'])
	data_in = tp[0]['b']
	model.regress(data_in - tp[1]['b'], i_in, i_out, reg=0.01)
	prod_1 = model.prodgmm(tp)

	### get state sequence ###
	# print model.PriorsR
	h_1 = model.forward_variable_priors(t, model.PriorsR, start_priors=model.StatesPriors)

	hmmr.create_distribution_fwd(h_1, start_pos=None)  # 64.3 ms  ~1.5 ms per timestep
	prod_ph_1 = hmmr.prodgmm(tp)

	lqr = pbd.LQR(canonical=True, horizon=70, rFactor=-2.0, nb_dim=3)

	q = np.argmax(h_1, axis=0)
	# print q
	# make a rest at the end
	q = np.concatenate([q, np.ones(20) * q[-1]])

	lqr.set_hmm_problem(prod_ph_1, range(50) + [49] * 20)
	lqr.evaluate_gains_infiniteHorizon()

	plan, command = lqr.solve_hmm_problem(start_point)

	if plot_on:
		label_size = 15
		idx = np.floor(np.linspace(1, 255, 50)).astype(int)
		color_gmr = cmap.viridis(range(256))[idx, 0:3]  # for states
		idx = np.floor(np.linspace(1, 255, nb_states)).astype(int)
		color = cmap.viridis(range(256))[idx, 0:3]  # for states
		fig = plt.figure(3 + demo_idx, figsize=(5, 5))
		# fig.suptitle("Reproduction", fontsize=14, fontweight='bold')
		nb_plt = 2
		ax = []  # subplots
		### specify subplots ###
		gs = gridspec.GridSpec(2, 1, width_ratios=[1], height_ratios=[4, 1])

		for j in [0, 1]:
			ax.append(fig.add_subplot(gs[j]))
		for a in ax:
			a.set_axis_bgcolor('white')

		### plot regressed HMM ###
		nb = 0  # subplots counter
		ax[nb].set_title(r'$\mathrm{(a)}$')

		for j in range(nb_samples):
			demo_plt, = ax[0].plot(demos[j]['Glb'][0, :], demos[j]['Glb'][1, :], 'k:', lw=1,
								   alpha=1)

		ax[nb].plot(data_in[0], data_in[1], '^', color=color[-1], ms=12)

		pblt.plot_gmm(prod_1.Mu, prod_1.Sigma, dim=[0, 1], color=color,
					  alpha=model.PriorsR * nb_states, ax=ax[nb], nb=2)

		### plot state sequence ###
		nb += 1

		for i in range(nb_states):
			ax[nb].plot(h_1[i, :], color=color[i])

		ax[nb].set_ylim([-0.1, 1.1])

		pblt.plot_gmm(prod_ph_1.Mu, prod_ph_1.Sigma, dim=[0, 1], color=color_gmr,
					  ax=ax[nb - 1],
					  nb=1)

		ax[0].plot(plan[0, :], plan[1, :], 'w', lw=2, zorder=50)
		ax[0].plot(demos[demo_idx]['Glb'][0, :], demos[demo_idx]['Glb'][1, :], 'k--', lw=3,
				   alpha=1, zorder=49)

	return np.copy(plan)
