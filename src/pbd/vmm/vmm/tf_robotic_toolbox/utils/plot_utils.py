import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def plot_robots(rob, qs, ax=None, color='k', xlim=None,ylim=None, feed_dict={},
				dx=0.02, dy=0.02, fontsize=10, text=True, bicolor=None, x_base=None, **kwargs):
	qs = np.array(qs).astype(np.float32)

	if qs.ndim == 1:
		qs = qs[None]

	if x_base is None:
		xs = rob.xs(tf.convert_to_tensor(qs)).eval(feed_dict)
	else:
		xs = rob.xs(tf.convert_to_tensor(qs), x_base=tf.convert_to_tensor(x_base)).eval(feed_dict)
	if text:
		for i in range(xs.shape[1]-1):
			plt.annotate(r'$q_%d$' % i,
						 (xs[0, i, 0], xs[0, i, 1]),
						 (xs[0, i, 0]+dx, xs[0, i, 1]+dy), fontsize=fontsize)

	if hasattr(rob, '_arms'):
		n_joint_arms = xs.shape[1]/2
		plot = ax if ax is not None else plt
		for i, x in enumerate(xs[:, :2]):
			plot.plot(x[:, 0], x[:, 1], marker='o', color='k', lw=10, mfc='w',
					 solid_capstyle='round', label='base_%d' % i,
					 **kwargs)

		for i, x in enumerate(xs[:, 1:int(n_joint_arms)]):
			plot.plot(x[:, 0], x[:, 1], marker='o', color=color, lw=10, mfc='w',
					 solid_capstyle='round', label='arm1_%d' % i,
					 **kwargs)

		for i, x in enumerate(xs[:, int(n_joint_arms)+1:]):
			bicolor = color if bicolor is None else bicolor
			plot.plot(x[:, 0], x[:, 1], marker='o', color=bicolor, lw=10, mfc='w',
					 solid_capstyle='round', label='arm2_%d' % i,
					 **kwargs)

	else:
		plot = ax if ax is not None else plt
		for x in xs:
			plot.plot(x[:, 0], x[:, 1], marker='o', color=color, lw=10, mfc='w',
					 solid_capstyle='round',
					 **kwargs)

	if ax is None:
		plt.axes().set_aspect('equal')
	else:
		ax.set_aspect('equal')

	if xlim is not None: plt.xlim(xlim)
	if ylim is not None: plt.ylim(ylim)

def plot_robot(xs, color='k', xlim=None,ylim=None, **kwargs):

	l = plt.plot(xs[:, 0], xs[:,1], marker='o', color=color, lw=10, mfc='w', solid_capstyle='round',
			 **kwargs)

	plt.axes().set_aspect('equal')

	if xlim is not None: plt.xlim(xlim)
	if ylim is not None: plt.ylim(ylim)

	return l

def plot_round_obstacle(x, r, n_segments=20, ax=None, **kwargs):
	x = np.array(x)
	t = np.linspace(-np.pi, np.pi, n_segments);
	points = np.array([np.cos(t), np.sin(t)]).T
	points = x[None] + r * points

	polygon = plt.Polygon(points, **kwargs)

	if ax is None:
		plt.gca().add_patch(polygon)
	else:
		ax.add_patch(polygon)