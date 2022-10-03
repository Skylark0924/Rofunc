import numpy as np
import os
import matplotlib.pyplot as plt
from .interactive import Interactive
from colorama import init  # need this to print colored on Windows and init() after
from termcolor import colored
import tkinter as tk
from tkinter.filedialog import asksaveasfilename
from matplotlib import gridspec


class Robot(object):
	def __init__(self, T):
		self.x, self.dx, self.dt = np.array([0., 0.]), np.array([0., 0.]), 1. / T
		self.ddx, self.fsx, self.fx = np.array([0., 0.]), np.array([0.]), np.array([0., 0.])
		self.sensor_mode = 0


class InteractiveDemos(Interactive, Robot):
	"""
	GUI for recording demonstrations in 2D
	"""

	def __init__(self, filename='test', path='', plot_function=None, **kwargs):
		Interactive.__init__(self)
		Robot.__init__(self, self.simulation_T)

		init()  # need to import colorama to use this, see imports

		self.path = os.path.dirname(__file__) if path == '' else path
		self.fig = plt.figure(figsize=(15, 8), facecolor='white')
		self.bindings.update({
			'q': (self.save_demos, [], "save demos"),
			'c': (self.clear_demos, [], "clear demos"),
			'x': (self.clear_demos, [True], "clear last demos"),
			'i': ([self.incr_param, self.highlight_demos], [['current_demo'], []], "next demos"),
			'd': (self.clear_demos, [False, True], "clear selected demos"),
		})

		gs = gridspec.GridSpec(1, 2)

		self.filename = filename

		self.ax_x = plt.subplot(gs[0])
		self.ax_dx = plt.subplot(gs[1])

		if plot_function:
			if type(plot_function) is list:
				for fnc in plot_function:
					fnc(self.ax_x)
			else:
				plot_function(self.ax_x)

		self.set_events()
		self.set_plots()

		self.is_demonstrating = False
		self.velocity_mode = False

		self.curr_demo, self.curr_demo_dx = [], []
		self.curr_demo_obj, self.curr_demo_obj_dx = [], []

		self._current_demo = {'x': [], 'dx': []}

		self.curr_mouse_pos = None
		self.robot_pos = np.zeros(2)

		self.nb_demos = 0
		self.demos = {'x': [], 'dx': []}

		self.params.update({'current_demo': [0, 0, self.nb_demos]})

		self.loaded = False
		#
		# win = plt.gcf().canvas.manager.window
		#
		# win.lift()
		# win.attributes("-topmost", True)
		# win.attributes("-alpha", 0.4)

		try:
			self.demos = np.load(self.path + filename + '.npy')[()]
			self.nb_demos = self.demos['x'].__len__();
			self.params['current_demo'][2] = self.nb_demos - 1
			print(colored('Existing skill, demos loaded', 'green'))
			self.replot_demos()
			self.fig.canvas.draw()
			self.loaded = True
		except:
			self.demos = {'x': [], 'dx': []}
			print(colored('Not existing skill', 'red'))

	def highlight_demos(self):
		data = self.demos['x'][self.params['current_demo'][0]]
		self.plots['current_demo'].set_data(data[:, 0], data[:, 1])
		self.fig.canvas.draw()

	def plot_sensor_value(self, s, scale=1.):
		data = np.vstack([self.x + np.array([0., 1.]) * s * scale, self.x])
		data -= 5. * np.array([0., 1.])[None]

		self.plots['sensor_value'].set_data(data[:, 0], data[:, 1])

	def set_plots(self):
		self.plots.update({
			'robot_plot': self.ax_x.plot([], [], 'o-', mew=4, mec='orangered', ms=10, mfc='w')[0],
			'sensor_value': self.ax_x.plot([], [], ls=(0, (1, 1)), lw=10)[0],
			'attractor_plot': self.ax_x.plot([], [], 'o-', mew=4, mec='teal', ms=10, mfc='w')[0],
			'obj_plot': self.ax_x.plot([], [], 'o-', mew=4, mec='steelblue', ms=10, mfc='w')[0],
			'current_demo': self.ax_x.plot([], [], lw=3, ls='--', color='orangered')[0],
			'current_demo_dx': self.ax_dx.plot([], [], lw=3, ls='--', color='orangered')[0]
		})

		for ax, lim in zip([self.ax_x, self.ax_dx], [10, 2.5]):  # [100, 25]
			ax.set_xlim([-lim, lim])
			ax.set_ylim([-lim, lim])

	def sim_dynamics(self, ffx, n_steps=10):
		if not self.velocity_mode:
			m = 1.0

			ddx = ffx / m
			self.x += self.dt / n_steps * self.dx + 0.5 * self.ddx * (
					self.dt / n_steps) ** 2
			self.dx += self.dt / n_steps * 0.5 * (self.ddx + ddx)
			self.dxx = np.copy(ddx)
		else:
			kp = 0
			kv = kp ** 0.5 * 2
			for i in range(50):
				ddx = kp * (self.curr_mouse_pos - self.dx)
				self.dx += self.dt * ddx
				self.x += self.dt * self.dx + (self.dt ** 2) / 2. * ddx

	def timer_event(self, event):
		if self.is_demonstrating:
			if self.curr_mouse_pos is None: self.pretty_print('Outside'); return

			# print self.x, self.dx
			kp = 400.
			kv = kp ** 0.5 * 2

			n_steps = 10
			for i in range(n_steps):
				ffx = kp * (self.curr_mouse_pos - self.x) - kv * self.dx
				self.sim_dynamics(ffx)

			# self.curr_demo += [np.copy(self.x)]; self.curr_demo_dx += [np.copy(self.dx)]

			self._current_demo['x'] += [np.copy(self.x)]
			self._current_demo['dx'] += [np.copy(self.dx)]

	def move_event(self, event):
		self.curr_mouse_pos = None if None in [event.xdata, event.ydata] else np.array([event.xdata, event.ydata])

		if event.key == 'shift' or self.is_demonstrating:
			self.robot_pos = np.copy(self.curr_mouse_pos)

			if not self.is_demonstrating:
				self.plots['robot_plot'].set_data(self.robot_pos[0], self.robot_pos[1])
				self.fig.canvas.draw()

	def plot_timer_event(self, event):
		self.robot_pos = self.curr_mouse_pos if self.robot_pos is None else self.robot_pos
		self.plots['attractor_plot'].set_data(self.robot_pos[0], self.robot_pos[1])
		self.plots['robot_plot'].set_data(self.x[0], self.x[1])

		if self.is_demonstrating:
			curr_demo_arr = np.array(self._current_demo['x'])
			curr_demo_dx_arr = np.array(self._current_demo['dx'])

			self.plots['current_demo'].set_data(curr_demo_arr[:, 0],
												curr_demo_arr[:, 1])

			self.plots['current_demo_dx'].set_data(curr_demo_dx_arr[:, 0],
												   curr_demo_dx_arr[:, 1])
			self.fig.canvas.draw()

	def click_event(self, event):
		if event.key is None:
			self.pretty_print('Demonstration started')
			self.velocity_mode = event.inaxes == self.ax_dx
			self.is_demonstrating = True
			if not self.velocity_mode:
				self.x = self.curr_mouse_pos
			else:

				self.x = self.demos['x'][-1][0] if self.nb_demos > 0 else np.array([0., 0.])
				self.dx = self.curr_mouse_pos

			[t.start() for t in [self.timer, self.plot_timer]]

	def release_event(self, event):
		if event.key is None:
			self.pretty_print('Demonstration finished')
			self.is_demonstrating = False
			self.finish_demo()

			[t.stop() for t in [self.timer, self.plot_timer]]

	def replot_demos(self):
		for i in range(self.nb_demos):
			data = self.demos['x'][i]
			self.plots['demo_%d' % i] = \
				self.ax_x.plot(data[:, 0], data[:, 1], lw=2, ls='--')[0]
			data = self.demos['dx'][i]
			self.plots['demo_dx_%d' % i] = \
				self.ax_dx.plot(data[:, 0], data[:, 1], lw=2, ls='--')[0]

	def clear_demos(self, last=False, selected=False):
		"""
		:param last: 	 [bool]
			Delete only last one
		"""
		if last or selected:
			idx = -1 if last else self.params['current_demo'][0]

			for s in self.demos:
				self.demos[s].pop(idx)

			for i in range(self.nb_demos):
				self.plots['demo_%d' % (i)].remove()
				self.plots['demo_dx_%d' % (i)].remove()

			self.nb_demos = len(self.demos['x']);
			self.params['current_demo'][2] = self.nb_demos - 1

			self.replot_demos()

			if selected:
				self.plots['current_demo'].set_data([], [])

			self.fig.canvas.draw()
		else:
			for i in range(self.nb_demos):
				self.plots['demo_%d' % i].remove()
				self.plots['demo_dx_%d' % i].remove()

			self.fig.canvas.draw()

			for s in self.demos:
				self.demos[s] = []
			self.nb_demos = 0;
			self.params['current_demo'][2] = self.nb_demos - 1

	def finish_demo(self):
		"""
		Called when finishing a demonstration to store the data
		:return:
		"""
		curr_demo_arr = np.array(self._current_demo['x'])
		curr_demo_dx_arr = np.array(self._current_demo['dx'])

		# self.demos['x'] += [curr_demo_arr]; self.demos['dx'] += [curr_demo_dx_arr]
		# self.curr_demo = []; self.curr_demo_dx = []

		for s in self._current_demo:
			self.demos[s] += [np.array(self._current_demo[s])]
			self._current_demo[s] = []

		self.plots['current_demo'].set_data([], []);
		self.plots['current_demo_dx'].set_data([], [])
		self.plots['demo_%d' % self.nb_demos] = self.ax_x.plot(curr_demo_arr[:, 0], curr_demo_arr[:, 1], lw=2, ls='--')[
			0]
		self.plots['demo_dx_%d' % self.nb_demos] = \
		self.ax_dx.plot(curr_demo_dx_arr[:, 0], curr_demo_dx_arr[:, 1], lw=2, ls='--')[0]

		self.nb_demos += 1;
		self.params['current_demo'][2] = self.nb_demos - 1

		self.fig.canvas.draw()

	def save_demos(self):
		"""
		Saving demonstrations with filename prompt
		:return:
		"""
		root = tk.Tk();
		root.withdraw()

		file_path = asksaveasfilename(initialdir=self.path, initialfile=self.filename + '.npy')

		self.pretty_print("Demonstrations saved as\n " + file_path)

		np.save(file_path, self.demos)

		pass
