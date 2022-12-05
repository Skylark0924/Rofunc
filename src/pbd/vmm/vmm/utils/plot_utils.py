import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy as sp

class DensityPlotter(object):
	def __init__(self, f, nb_sub=20):
		self.x_batch = tf.compat.v1.placeholder(tf.float32, (nb_sub ** 2, 2))
		self.fx_batch = f(self.x_batch)

		self.nb_sub = nb_sub

	def f_np(self, x, feed_dict={}):
			feed_dict[self.x_batch] = x
			return self.fx_batch.eval(feed_dict)

	def plot(self, ax0=None, xlim=[-1, 1], ylim=[-1, 1], cmap='viridis',
				 lines=True, heightmap=True, inv=False, vmin=None, vmax=None, use_tf=False,
				 ax=None, feed_dict={}, exp=True, img=False, kwargs_lines={}):

		return plot_density(lambda x: self.f_np(x, feed_dict=feed_dict),
			nb_sub=self.nb_sub, ax0=ax0, xlim=xlim, ylim=ylim, cmap=cmap,
			lines=lines, heightmap=heightmap, inv=inv, vmin=vmin, vmax=vmax, use_tf=False,
			ax=ax, feed_dict=feed_dict, exp=exp, img=img, kwargs_lines=kwargs_lines)



def plot_density(f, nb_sub=10, ax0=None, xlim=[-1, 1], ylim=[-1, 1], cmap='viridis',
				 lines=True, heightmap=True, inv=False, vmin=None, vmax=None, use_tf=False,
				 ax=None, feed_dict={}, exp=True, img=False, kwargs_lines={}):

	if use_tf:
		x_batch = tf.compat.v1.placeholder(tf.float32, (nb_sub ** 2, 2))
		fx_batch = f(x_batch)
		def f_np(x):
			feed_dict[x_batch] = x
			return fx_batch.eval(feed_dict)

		f = f_np


	x = np.linspace(*xlim, num=nb_sub)
	y = np.linspace(*ylim, num=nb_sub)

	if img:
		Y, X = np.meshgrid(x, y)
	else:
		X, Y = np.meshgrid(x, y)

	zs = f(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
	Z = zs.reshape(X.shape)


	if lines:
		if ax is None:
			plt.contour(X, Y, Z, **kwargs_lines)
		else:
			ax.contour(X, Y, Z, **kwargs_lines)


	if heightmap:
		if exp:
			_img = np.exp(Z) if not inv else -np.exp(Z)
		else:
			_img = Z if not inv else -Z

		kwargs = {'origin': 'lower'}

		if ax is None:
			plt.imshow(_img, interpolation='bilinear', extent=xlim + ylim,
				   alpha=0.5, cmap=cmap, vmax=vmax, vmin=vmin, **kwargs)
		else:
			ax.imshow(_img, interpolation='bilinear', extent=xlim + ylim,
				   alpha=0.5, cmap=cmap, vmax=vmax, vmin=vmin, **kwargs)

	return Z

def plot_gmm(loc, cov, dim=None, color=[1, 0, 0], alpha=0.5, linewidth=1, markersize=6,
			 ax=None, empty=False, edgecolor=None, edgealpha=None,
			 border=False, nb=1, center=True, zorder=20, equal=True):


	if loc.ndim == 1:
		loc = loc[None]
	if cov.ndim == 2:
		cov = cov[None]

	nb_states = loc.shape[0]

	if dim:
		loc = loc[:, dim]
		cov = cov[np.ix_(range(cov.shape[0]), dim, dim)] if isinstance(dim, list) else cov[:, dim, dim]

	nbDrawingSeg = 20
	t = np.linspace(-np.pi, np.pi, nbDrawingSeg)

	if not isinstance(color, list) and not isinstance(color, np.ndarray):
		color = [color] * nb_states
	elif not isinstance(color[0], basestring) and not isinstance(color, np.ndarray):
		color = [color] * nb_states

	if not isinstance(alpha, np.ndarray):
		alpha = [alpha] * nb_states
	else:
		alpha = np.clip(alpha, 0.1, 0.9)

	for i, c, a in zip(range(0, nb_states), color, alpha):
		# Create Polygon

		R = np.real(sp.linalg.sqrtm(1.0 * cov[i]))
		points = R.dot(np.array([[np.cos(t)], [np.sin(t)]]).reshape([2, nbDrawingSeg])) + \
				 loc[[i]].T

		if edgecolor is None:
			edgecolor = c

		polygon = plt.Polygon(points.transpose().tolist(), facecolor=c, alpha=a,
							  linewidth=linewidth, zorder=zorder, edgecolor=edgecolor)

		if edgealpha is not None:
			plt.plot(points[0,:], points[1,:], color=edgecolor)

			# Set properties
		# polygon.set_alpha(0.3)
		# polygon.set_color(color)
		if ax:
			ax.add_patch(polygon)  # Patch

			l = None
			if center:
				a = alpha[i]
			else:
				a = 0.

			ax.plot(loc[i, 0], loc[i, 1], '.', color=c, alpha=a)  # Mean

			if border:
				ax.plot(points[0, :], points[1, :], color=c, linewidth=linewidth,
						markersize=markersize)  # Contour
			if equal:
				ax.set_aspect('equal')
		else:
			if empty:
				plt.gca().grid('off')
				# ax[-1].set_xlabel('x position [m]')
				plt.gca().set_axis_bgcolor('w')
				plt.axis('off')

			plt.gca().add_patch(polygon)  # Patch
			l = None

			if center:
				a = alpha[i]
			else:
				a = 0.0

			l, = plt.plot(loc[i, 0], loc[i, 1], '.', color=c, alpha=a)  # Mean
					# plt.plot(points[0,:], points[1,:], color=c, linewidth=linewidth , markersize=markersize) # Contour
			if equal:
				plt.axes().set_aspect('equal')
	return l
