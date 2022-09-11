import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from scipy.stats import multivariate_normal
from .functions import *
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.cm as cm
import itertools

flatui = (1./255 * np.array([[51, 77, 92],  # kuler theme Flat design color
								  [69, 178, 157],
								  [239, 201, 76],
								  [226, 122, 63],
								  [226, 0, 0],
								  [255, 255, 255]])).tolist()

def plot_data(data, dim=[[0, 1]], figsize=(3, 3), fig=None):
	if not isinstance(dim[0], list):
		dim = [dim]

	nb_plt = len(dim)
	if fig is None:
		fig = plt.figure(figsize=(figsize[0] * nb_plt, figsize[1]))
	ax = []
	label_size = 15

	nb = 0
	for j in range(nb_plt):
		ax.append(fig.add_subplot(1, nb_plt, j + 1))
	# ax[-1].grid('off')
	# ax[-1].set_xlabel('x position [m]')

	# ax[-1].set_axis_bgcolor('w')

	for i, di in enumerate(dim):
		ax[i].plot(*[data[:, i] for i in di])

	return ax


def plot_distpatch(ax, x, mean, var, color=[1, 0, 0], num_std=2, alpha=0.5, linewidth=1,
				   linealpha=1):
	''' Function plots the mean and corresponding variance onto the specified axis

	ax : axis object where the distribution patch should be plotted
	X  : nbpoints array of x-axis values
	Mu : nbpoints array of mean values corresponding to the x-axis values
	Var: nbpoints array of variance values corresponding to the x-axis values

	Author: Martijn Zeestrate, 2015
	'''

	# get number of points:
	npoints = len(x)

	# Create vertices:
	xmsh = np.append(x, x[::-1])
	vTmp = np.sqrt(var + 1e-6) * num_std
	ymsh = np.append(vTmp + mean, mean[::-1] - vTmp[::-1])
	msh = np.concatenate((xmsh.reshape((2 * npoints, 1)), ymsh.reshape((2 * npoints, 1))),
						 axis=1)
	msh = np.concatenate((msh, msh[-1, :].reshape((1, 2))), axis=0)

	# Create codes
	codes = [Path.MOVETO]
	codes.extend([Path.LINETO] * (2 * npoints - 1))
	codes.extend([Path.CLOSEPOLY])

	# Create Path
	path = Path(msh, codes)
	patch = patches.PathPatch(path, facecolor=color, lw=0, edgecolor=color, alpha=alpha)

	# Add to axis:
	ax.add_patch(patch)  # Patch
	ax.plot(x, mean, linewidth=linewidth, color=color, alpha=linealpha)  # Mean

def plot_spherical_gmm(Mu, Sigma, dim=None, tp=None, color='r',
						   alpha=255, swap=False, ax=None, label=None):
		"""

		:param Mu:
		:param Sigma:
		:param dim:
		:param tp:
		:param color: Tuple (R, G, B) 0-255, Tuple (R, G, B, A), np.array((Nx3))
		:param alpha:
		:return:
		"""

		if isinstance(Mu, float):
			Mu = np.array(Mu)[None]
			Sigma = np.array(Sigma)[None]
			nbStates = 1

		elif (Mu.ndim == 1):
			Mu = Mu[:, np.newaxis]
			Sigma = Sigma[:, :, np.newaxis]
			nbStates = 1
		else:
			if not swap:
				nbStates = Mu.shape[1]
				nbVar = Mu.shape[0]
			else:
				nbStates = Mu.shape[0]
				nbVar = Mu.shape[1]

		if dim:
			if swap:
				Mu = Mu[:, dim]
				sl = np.ix_(range(nbStates), dim, dim)
				Sigma = Sigma[sl]
			else:
				Mu = Mu[dim, :]
				sl = np.ix_(dim, dim)
				Sigma = Sigma[sl]


		# if (Mu.ndim == 1):
		# 	nbVar = Mu.shape[0]
		# 	nbStates = 1
		# 	# Fix size of arrays:
		# 	Mu = Mu.reshape(1, nbVar)
		# 	Sigma = Sigma.reshape(1, nbVar, nbVar)
		# else:
		# 	nbStates = Mu.shape[1]
		# 	nbVar = Mu.shape[0]
		#
		# if dim:
		# 	Mu = Mu[dim, :]
		# 	sl = np.ix_(dim, dim)
		# 	Sigma = Sigma[sl]

		nbDrawingSeg = 35

		t = np.linspace(-np.pi, np.pi, nbDrawingSeg)

		if not isinstance(color, list) and not isinstance(color, np.ndarray):
			color = [color] * nbStates

		if not isinstance(alpha, np.ndarray):
			alpha = alpha * np.ones(nbStates)
		else:
			alpha = np.clip(alpha, 50, 255)

		for i, col in zip(range(nbStates), color):

			# TODO plot variance
			pts = (np.array([[np.cos(t)], [np.sin(t)]]).reshape([2, nbDrawingSeg]))
			points_ext = (Mu[i] + Sigma[i]**0.5) * pts


			points = (Mu[i]) * pts

			points_int = (Mu[i] - Sigma[i]**0.5	) * pts

			if tp:
				points += tp['b'][:, np.newaxis]
				points_ext += tp['b'][:, np.newaxis]
				points_int += tp['b'][:, np.newaxis]

			if len(col) == 3:  # if alpha not already defined
				if isinstance(col, np.ndarray):
					c = np.append(col, alpha[i])
				else:
					c = col + (alpha[i],)
			else:
				c = col

			if ax is None:
				p, a = (plt, plt.axes())
			else:
				p, a = (ax, ax)
			_label_std = None if label is None else label + ' std'
			p.plot(points_int[0, :], points_int[1, :], lw=1, alpha=1, color=col, ls='--')
			p.plot(points_ext[0, :], points_ext[1, :], lw=1, alpha=1, color=col, ls='--')
			p.plot(points[0, :], points[1, :], lw=1, alpha=1, color=col, label=label)

			# plt.fill_between(points_ext[0], points_int[1], points_ext[1])


def plot_coordinate_system(A, b, scale=1., equal=True, ax=None, **kwargs):
	"""

	:param A:		nb_dim x nb_dim
		Rotation matrix
	:param b: 		nb_dim
		Translation
	:param scale: 	float
		Scaling of the axis
	:param equal: 	bool
		Set matplotlib axis to equal
	:param ax: 		plt.axes()
	:param kwargs:
	:return:
	"""
	a0 = np.vstack([b, b + scale * A[:,0]])
	a1 = np.vstack([b, b + scale * A[:,1]])

	if ax is None:
		p, a = (plt, plt.axes())
	else:
		p, a = (ax, ax)

	if equal:
		a.set_aspect('equal')

	p.plot(a0[:, 0], a0[:, 1], 'r', **kwargs)
	p.plot(a1[:, 0], a1[:, 1], 'b', **kwargs)

def plot_linear_system(K, b=None, name=None, nb_sub=10, ax0=None, xlim=[-1, 1], ylim=[-1, 1],
					   equal=True, scale=0.01, scale_K=100, plot_gains=True, field=None, multi_center=False, **kwargs):
	a = -2


	Y, X = np.mgrid[ylim[0]:ylim[1]:complex(nb_sub), xlim[0]:xlim[1]:complex(nb_sub)]
	mesh_data = np.vstack([X.ravel(), Y.ravel()])

	plot_center = True
	if b is None:
		b = np.zeros(2); plot_center = False

	field = np.einsum('ij,ja->ia', K, mesh_data-b[:,None]) if field is None else field

	# if plot_gains:
	# 	args = {}
	# 	if ax0 is not None:
	# 		args['ax'] =  ax0
	# 	if 'color' in kwargs:
	# 		args['color'] = kwargs['color']

		# pbd.plot_gmm(b, - scale_K* np.linalg.inv(K), **args)

	U = field[0]
	V = field[1]
	U = U.reshape(nb_sub, nb_sub)
	V = V.reshape(nb_sub, nb_sub)
	# import pdb; pdb.set_trace()
	speed = np.sqrt(U * U + V * V)


	if name is not None:
		plt.suptitle(name)

	if ax0 is not None:
		strm = ax0.streamplot(X, Y, U, V, linewidth=scale* speed,**kwargs)
		ax0.set_xlim(xlim)
		ax0.set_ylim(ylim)

		if plot_center:
			if multi_center: ctr = ax0.plot(b[:, 0], b[:, 1], 'kx', ms=8, mew=2)[0]
			else: ctr = ax0.plot(b[0], b[1], 'kx', ms=8, mew=2)[0]

		if equal:
			ax0.set_aspect('equal')

	else:
		strm = plt.streamplot(X, Y, U, V, linewidth=scale* speed, **kwargs)
		plt.xlim(xlim)
		plt.ylim(ylim)

		if plot_center:
			if multi_center: ctr = plt.plot(b[:, 0], b[:, 1], 'kx', ms=8, mew=2)[0]
			else: ctr = plt.plot(b[0], b[1], 'kx', ms=8, mew=2)[0]

		if equal:
			plt.axes().set_aspect('equal')

	if plot_center:
		return [strm, ctr]
	else:
		return [strm]


def plot_function_map(f, nb_sub=10, ax=None, xlim=[-1, 1], ylim=[-1, 1], opp=False, exp=False, vmin=None, vmax=None, contour=True):
	"""

	:param f:			[function]
	 	A function to plot that can take an array((N, nb_dim)) as input
	:param nb_sub:
	:param ax0:
	:param xlim:
	:param ylim:
	:return:
	"""
	x = np.linspace(*xlim)
	y = np.linspace(*ylim)
	xx, yy = np.meshgrid(x, y)

	# Y, X = np.mgrid[ylim[0]:ylim[1]:complex(nb_sub), xlim[0]:xlim[1]:complex(nb_sub)]
	mesh_data = np.concatenate([np.atleast_2d(xx.ravel()), np.atleast_2d(yy.ravel())]).T
	try:
		zz = f(mesh_data)
	except: # if function cannot take a vector as input
		zz = np.array([f(_x) for _x in mesh_data])

	z = zz.reshape(xx.shape)

	if ax is None:
		ax = plt

	if contour:
		try:
			CS = ax.contour(xx, yy, z, cmap='viridis')
			ax.clabel(CS, inline=1, fontsize=10)
		except:
			pass
	if opp: z = -z
	if exp: z = np.exp(z)
	ax.imshow(z, interpolation='bilinear', origin='lower', extent=xlim + ylim,
			   alpha=0.5, cmap='viridis', vmin=vmin, vmax=vmax)

	return np.min(z), np.max(z)

def plot_mixture_linear_system(model, mode='glob', nb_sub=20, gmm=True, min_alpha=0.,
							   cmap=plt.cm.jet, A=None,b=None, gmr=False, return_strm=False,
							   **kwargs):
	"""

	:param model:
	:param mode: 		in ['glob', 'glob_overlay', 'local']
	:param nb_sub:
	:param min_alpha:
	:param cmap:
	:param kwargs:
	:return:
	"""
	from matplotlib.colors import ListedColormap

	xlim, ylim = [kwargs.get(s, [0, 1]) for s in ['xlim', 'ylim']]


	Y, X = np.mgrid[ylim[0]:ylim[1]:complex(nb_sub), xlim[0]:xlim[1]:complex(nb_sub)]
	mesh_data = np.vstack([X.ravel(), Y.ravel()])
	statecmap = cmap(range(256))[np.rint(np.linspace(0, 255, model.nb_states)).astype(int),:-1]

	if mode == 'glob':
		if gmr:
			field = model.condition(mesh_data.T, dim_in=slice(0, 2), dim_out=slice(2, 4))[0]
			model.center = None
		else:
			field = model.condition(mesh_data.T) if A is None else model.condition(mesh_data.T, A, b)
		strm = plot_linear_system(None, model.center, field=field.T, nb_sub=nb_sub, multi_center=hasattr(model, 'unconstrained_center'),
						   **kwargs)

	else:
		# compute responsabilities
		l = np.zeros((mesh_data.shape[1], model.nb_states))
		for i in range(model.nb_states):
			l[:, i] = multi_variate_normal(mesh_data.T, model.mus_in[i], model.sigmas_in[i])

		l += np.log(model.priors)[None]

		l = np.exp(l)
		gamma = l / np.sum(l, axis=1)[:, None]

		# create color map, blending to one color to transparent for each states
		cmaps = [np.ones((256, 4)) for i in range(model.nb_states)]


		# create each cmaps from color of an overall cmap. Each cmaps blends from this color to transparent.
		for i in range(model.nb_states):
			cmaps[i][:, :3] = statecmap[i]
			cmaps[i][:, 3] = np.linspace(min_alpha, 1, cmap.N)

		cmaps = [ListedColormap(cmap_) for cmap_ in cmaps]

		colors = [gamma[:, i].reshape(nb_sub, nb_sub) for i in range(model.nb_states)]

		for i in range(model.nb_states):
			if mode == 'glob_overlay':
				field = model.condition(mesh_data.T)
				plot_linear_system(None, model.center, field=field.T	, color=colors[i], cmap=cmaps[i],
								   nb_sub=nb_sub, **kwargs)
			else:
				plot_linear_system(model.ks[i], model.center, color=colors[i], cmap=cmaps[i],
								   nb_sub=nb_sub, **kwargs)

	if gmm:
		if gmr:
			mu, sigma = model.mu[:, :2], model.sigma[:, :2, :2]
		else:
			if A is None:
				mu, sigma = (model.mus_in, model.sigmas_in)#
			else:
				mu, sigma = (model.mus_in_p[:, 0], model.sigmas_in_p[:, 0])

		plot_gmm(mu, sigma, swap=True, color=statecmap, ax=kwargs.pop('ax0', None), zorder=0)

	if return_strm:
		return statecmap, strm
	else:
		return statecmap


def plot_gmm(Mu, Sigma, dim=None, color=[1, 0, 0], alpha=0.5, linewidth=1, markersize=6,
			 ax=None, empty=False, edgecolor=None, edgealpha=None, priors=None,
			 border=False, nb=1, swap=True, center=True, zorder=20):
	''' This function displays the parameters of a Gaussian Mixture Model (GMM).

	 Inputs -----------------------------------------------------------------
	   o Mu:           D x K array representing the centers of K Gaussians.
	   o Sigma:        D x D x K array representing the covariance matrices of K Gaussians.

	 Author:    Martijn Zeestraten, 2015
			 http://programming-by-demonstration.org/martijnzeestraten

			 Note- Daniel Berio, switched matrix layout to be consistent with pbdlib matlab,
				   probably breaks with gmm now.
	'''
	Mu = np.array(Mu)
	Sigma = np.array(Sigma)
	if (Mu.ndim == 1):
		if not swap:
			Mu = Mu[:, np.newaxis]
			Sigma = Sigma[:, :, np.newaxis]
			nbStates = 1
		else:
			Mu = Mu[np.newaxis]
			Sigma = Sigma[np.newaxis]
			nbStates = 1
	else:
		if not swap:
			nbStates = Mu.shape[1]
			nbVar = Mu.shape[0]
		else:
			nbStates = Mu.shape[0]
			nbVar = Mu.shape[1]

	if dim:
		if swap:
			Mu = Mu[:, dim]
			sl = np.ix_(range(nbStates),dim, dim)
			Sigma = Sigma[sl]
		else:
			Mu = Mu[dim, :]
			sl = np.ix_(dim, dim)
			Sigma = Sigma[sl]

	if priors is not None:
		priors /= np.max(priors)
		priors = np.clip(priors, 0.1, 1.)

	nbDrawingSeg = 35;
	t = np.linspace(-np.pi, np.pi, nbDrawingSeg);

	if not isinstance(color, list) and not isinstance(color, np.ndarray):
		color = [color] * nbStates
	elif not isinstance(color[0], str) and not isinstance(color, np.ndarray):
		color = [color] * nbStates

	if not isinstance(alpha, np.ndarray):
		alpha = [alpha] * nbStates
	else:
		alpha = np.clip(alpha, 0.1, 0.9)

	for i, c, a in zip(range(0, nbStates), color, alpha):
		# Create Polygon
		if not swap:
			R = np.real(sp.linalg.sqrtm(1.0 * Sigma[:, :, i]))
			points = R.dot(
				np.array([[np.cos(t)], [np.sin(t)]]).reshape([2, nbDrawingSeg])) + Mu[:,
																				   i].reshape(
				[2, 1])
		else:
			R = np.real(sp.linalg.sqrtm(1.0 * Sigma[i]))
			points = R.dot(np.array([[np.cos(t)], [np.sin(t)]]).reshape([2, nbDrawingSeg])) + \
					 Mu[[i]].T

		if edgecolor is None:
			edgecolor = c

		if priors is not None: a *= priors[i]

		polygon = plt.Polygon(points.transpose().tolist(), facecolor=c, alpha=a,
							  linewidth=linewidth, zorder=zorder, edgecolor=edgecolor)

		if edgealpha is not None:
			plt.plot(points[0,:], points[1,:], color=edgecolor)

		if nb == 2:
			R = np.real(sp.linalg.sqrtm(4.0 * Sigma[:, :, i]))
			points = R.dot(
				np.array([[np.cos(t)], [np.sin(t)]]).reshape([2, nbDrawingSeg])) + Mu[:,
																				   i].reshape(
				[2, 1])
			polygon_2 = plt.Polygon(points.transpose().tolist(), facecolor=c, alpha=a / 2.,
									linewidth=linewidth, zorder=zorder-5, edgecolor=edgecolor)
			# Set properties
		# polygon.set_alpha(0.3)
		# polygon.set_color(color)
		if ax:
			if nb == 2:
				ax.add_patch(polygon_2)
			ax.add_patch(polygon)  # Patch

			l = None
			if center:
				a = alpha[i]
			else:
				a = 0.

			if not swap:
				ax.plot(Mu[0, i], Mu[1, i], '.', color=c, alpha=a)  # Mean
			else:
				ax.plot(Mu[i, 0], Mu[i, 1], '.', color=c, alpha=a)  # Mean

			if border:
				ax.plot(points[0, :], points[1, :], color=c, linewidth=linewidth,
						markersize=markersize)  # Contour
		else:
			if empty:
				plt.gca().grid('off')
				# ax[-1].set_xlabel('x position [m]')
				plt.gca().set_axis_bgcolor('w')
				plt.axis('off')

			plt.gca().add_patch(polygon)  # Patch
			if nb == 2:
				ax.add_patch(polygon_2)
			l = None

			if center:
				a = alpha[i]
			else:
				a = 0.0

			if not swap:
				l, = plt.plot(Mu[0, i], Mu[1, i], '.', color=c, alpha=a)  # Mean
			else:
				l, = plt.plot(Mu[i, 0], Mu[i, 1], '.', color=c, alpha=a)  # Mean

			if border:
				plt.plot(points[0, :], points[1, :], color=c, linewidth=linewidth,
						markersize=markersize)  # Contour
					# plt.plot(points[0,:], points[1,:], color=c, linewidth=linewidth , markersize=markersize) # Contour

	return l

def plot_gaussian(mu, sigma, dim=None, color='r', alpha=0.5, lw=1, markersize=6,
			 ax=None, plots=None, nb_segm=24, **kwargs):

	mu, sigma = np.array(mu), np.array(sigma)

	t = np.linspace(-np.pi, np.pi, nb_segm)
	R = np.real(sp.linalg.sqrtm(1.0 * sigma))

	points = np.einsum('ij,ja->ia', R, np.array([np.cos(t), np.sin(t)])) + mu[:, None]

	if plots is None:
		p, a = (plt, plt.axes()) if ax is None else (ax, ax)
		center, = p.plot(mu[0], mu[1], '.', color=color, alpha=alpha)  # Mean

		line, =	p.plot(points[0], points[1], color=color, linewidth=lw,
					markersize=markersize, **kwargs)  # Contour
	else:
		center, line = plots
		center.set_data(mu[0], mu[1])
		line.set_data(points[0], points[1])


	return center, line

def plot_y_gaussian(x, mu, sigma, dim=0, alpha=1., alpha_fill=None, color='r', lw=1.,
					ax=None, label=None):
	"""

	:param mu: 		[n_states]
	:param mu: 		[n_states, n_dim]
	:param sigma: 	[n_states, n_dim, n_dim]
	:param dim:
	:return:
	"""
	if x.ndim == 2:
		x = x[:, 0]

	if alpha_fill is None:
		alpha_fill = 0.4 * alpha

	if ax is None:
		ax = plt

	ax.plot(x, mu[:, dim], alpha=alpha, color=color, label=label)
	ax.fill_between(x,
					 mu[:, dim] - sigma[:, dim, dim] ** 0.5,
					 mu[:, dim] + sigma[:, dim, dim] ** 0.5,
					 alpha=alpha_fill, color=color)

def plot_dynamic_system(f, nb_sub=10, ax=None, xlim=[-1, 1], ylim=[-1, 1], scale=0.01,
						name=None, equal=False, **kwargs):
	"""
	Plot a dynamical system dx = f(x)
	:param f: 		a function that takes as input x as [N,2] and return dx [N, 2]
	:param nb_sub:
	:param ax0:
	:param xlim:
	:param ylim:
	:param scale:
	:param kwargs:
	:return:
	"""

	Y, X = np.mgrid[ylim[0]:ylim[1]:complex(nb_sub), xlim[0]:xlim[1]:complex(nb_sub)]
	mesh_data = np.vstack([X.ravel(), Y.ravel()])

	field = f(mesh_data.T).T

	U = field[0]
	V = field[1]
	U = U.reshape(nb_sub, nb_sub)
	V = V.reshape(nb_sub, nb_sub)
	speed = np.sqrt(U * U + V * V)


	if name is not None:
		plt.suptitle(name)

	if ax is not None:
		strm = ax.streamplot(X, Y, U, V, linewidth=scale* speed,**kwargs)
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)

		if equal:
			ax.set_aspect('equal')

	else:
		strm = plt.streamplot(X, Y, U, V, linewidth=scale* speed, **kwargs)
		plt.xlim(xlim)
		plt.ylim(ylim)

		if equal:
			plt.axes().set_aspect('equal')

	return [strm]

def plot_trans(mu, trans, dim=[0, 1], a=0.1, ds=0.2, min_alpha=0.05, ax=None, **kwargs):
	std_mu = np.std(mu)
	kwargs['fc'] = kwargs.pop('fc', 'k')
	kwargs['ec'] = kwargs.pop('ec', 'k')
	kwargs['head_width'] = kwargs.pop('head_width', 0.2 * std_mu)
	kwargs['width'] = kwargs.pop('width', 0.03 * std_mu)
	trans_wd = trans - trans * np.eye(trans.shape[0])  # remove diag
	trans_wd /= (np.sum(trans_wd, axis=1, keepdims=True) + 1e-10)
	mu = mu[:, dim]


	for i in range(mu.shape[0]):
		for j in range(mu.shape[0]):
			if i == j: continue;
			alpha = (trans_wd[i, j] + min_alpha)/(1. + min_alpha)

			s = a  * mu[j] + (1.-a) * mu[i]
			e = (1.- a)  * mu[j] + (a) * mu[i]

			ortho = np.roll(e - s, 1) * np.array([-1., 1])
			ortho /= np.linalg.norm(ortho)

			s += ortho * ds * kwargs['head_width']
			e += ortho * ds * kwargs['head_width']

			d = e - s
			if ax is None:
				ax = plt
			ax.arrow(
				s[0], s[1], d[0], d[1], length_includes_head=True,
				alpha=alpha, shape='right', **kwargs)


def plot_trajdist(td, ix=0, iy=1, covScale=1, color=[1, 0, 0], alpha=0.1, linewidth=0.1):
	'''Plot 2D representation of a trajectory distribution'''

	meanM = td.mean.reshape(td.n_data, td.n_vars)
	expSigma = np.zeros((td.n_data, td.n_vars, td.n_vars))
	for i in range(0, td.n_data):
		st = i * td.n_vars
		end = (i + 1) * td.n_vars
		# Add regularization term to avoid warnings about singularities
		expSigma[i, :, :] = td.covar[st:end, st:end] * covScale + np.eye(td.n_vars) * 1e-5

	sel = np.ix_(np.arange(td.n_data), [ix, iy], [ix, iy])
	l = plot_gmm(meanM[:, (ix, iy)], expSigma[sel],
				 color=color, alpha=alpha, linewidth=linewidth)
	return l  # return handle


def plot_trajreference(meanQ, covarQ, n_vars, q, ax=[], colormap=[]):
	''' Plot Reference of GMM-based Trajectory Distributions. '''

	if ax != []:
		# check if size is correct
		if len(ax) < n_vars:
			raise Exception('not enough axis specified')
	else:
		# generate axis
		ax = []
		for i in range(n_vars):
			ax.append(plt.subplot(n_vars, 1, i + 1))

	n_components = q.max() + 1

	if colormap == []:
		colormap = cm.Set1(np.linspace(0, 1, n_components))[:, 0:3]

	x = np.arange(len(q))
	variance = np.diag(covarQ)

	for i in range(0, n_vars):
		tmp_var = variance[i::n_vars]
		tmp_mean = meanQ[i::n_vars]
		for j in range(0, n_components):
			if ((q == j).sum() > 1):
				print(j)
				# Patch:
				plot_distpatch(ax[i], x[q == j], tmp_mean[q == j],
							   tmp_var[q == j], color=colormap[j,],
							   alpha=0.2, linewidth=2)
	return ax


def plot_TP(TP, color=[0, 1, 0], alpha=1, scale=0.2):
	A = TP['A'] * scale
	b = TP['b']

	# Arrows:
	for i in range(len(A)):
		# Both lines and arrows are plotted.
		# The lines are needed to make sure that the shape of the frame is
		# automatically taken into account when calculating axis limits
		plt.plot([b[0], b[0] + A[0, i]], [b[1], b[1] + A[1, i]],
				 color=color, linewidth=3, alpha=alpha)
		plt.arrow(b[0], b[1], A[0, i], A[1, i], color=color, linewidth=3, alpha=alpha)

	# Origin:
	plt.plot(b[0], b[1], '.', markersize=20, color=color, alpha=alpha)


def periodic_clip(val, n_min, n_max):
	''' keeps val within the range [n_min, n_max) by assuming that val is a periodic value'''
	if val < n_max and val >= n_min:
		val = val
	elif val >= n_max:
		val = val - (n_max - n_min)
	elif val < n_max:
		val = val + (n_max - n_min)

	return val


def tri_elipsoid(n_rings, n_points):
	''' Compute the set of triangles that covers a full elipsoid of n_rings with n_points per ring'''
	tri = []
	for n in range(n_points - 1):
		# Triange down
		#       *    ring i+1
		#     / |
		#    *--*    ring i
		tri_up = np.array([n, periodic_clip(n + 1, 0, n_points),
						   periodic_clip(n + n_points + 1, 0, 2 * n_points)])
		# Triangle up
		#    *--*      ring i+1
		#    | /
		#    *    ring i

		tri_down = np.array([n, periodic_clip(n + n_points + 1, 0, 2 * n_points),
							 periodic_clip(n + n_points, 0, 2 * n_points)])

		tri.append(tri_up)
		tri.append(tri_down)

	tri = np.array(tri)
	trigrid = tri
	for i in range(1, n_rings - 1):
		trigrid = np.vstack((trigrid, tri + n_points * i))

	return np.array(trigrid)


def plot_gauss3d(ax, mean, covar, n_points=30, n_rings=20, color='red', alpha=0.3,
				 linewidth=0):
	''' Plot 3d Gaussian'''

	# Compute eigen components:
	(D0, V0) = np.linalg.eig(covar)
	U0 = np.real(V0.dot(np.diag(D0) ** 0.5))

	# Compute first rotational path
	psi = np.linspace(0, np.pi * 2, n_rings, endpoint=True)
	ringpts = np.vstack((np.zeros((1, len(psi))), np.cos(psi), np.sin(psi)))

	U = np.zeros((3, 3))
	U[:, 1:3] = U0[:, 1:3]
	ringtmp = U.dot(ringpts)

	# Compute touching circular paths
	phi = np.linspace(0, np.pi, n_points)
	pts = np.vstack((np.cos(phi), np.sin(phi), np.zeros((1, len(phi)))))

	xring = np.zeros((n_rings, n_points, 3))
	for j in range(n_rings):
		U = np.zeros((3, 3))
		U[:, 0] = U0[:, 0]
		U[:, 1] = ringtmp[:, j]
		xring[j, :] = (U.dot(pts).T + mean)

	# Reshape points in 2 dimensional array:
	points = xring.reshape((n_rings * n_points, 3))

	# Compute triangle points:
	triangles = tri_elipsoid(n_rings, n_points)

	# Plot surface:
	ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
					triangles=triangles, linewidth=linewidth, alpha=alpha, color=color,
					edgecolor=color)


def plot_gmm3d(ax, means, covars, n_points=20, n_rings=15, color='red', alpha=0.4,
			   linewidth=0):
	''' Plot 3D gmm '''
	n_states = means.shape[0]
	for i in range(n_states):
		print
		plot_gauss3d(ax, means[i,], covars[i,],
					 n_points=n_points, n_rings=n_rings, color=color,
					 alpha=alpha, linewidth=linewidth)


def plot_gaussian1d(Mu, Sigma, lim=[-1, 1], color=[1, 0, 0], alpha=0.5, lw=1, ax=None, nb=1,
					nb_step=100):
	x = np.linspace(lim[0], lim[1], nb_step)
	y = multivariate_normal.pdf(x, Mu, Sigma)

	plt.plot(x, y, lw=lw, c=color, alpha=alpha)
	plt.fill_between(x, y, color=color, alpha=alpha * 0.2)

	return x, y
