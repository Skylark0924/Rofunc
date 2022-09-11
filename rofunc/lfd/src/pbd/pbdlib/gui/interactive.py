import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)

try:
	from termcolor import colored
except:
	def colored(*args, **kwargs):
		return args

plt.style.use('ggplot')

class Interactive(object):
	def __init__(self):
		self.bindings = {}
		self.plots = {}
		self.params = {}
		self.simulation_T = 20

	def key_event(self, event):
		c = event.key
		print(c)
		if c in ['\x1b', '\x03']:
			done = True
		elif c in self.bindings:
			cmd = self.bindings[c]
			if isinstance(cmd[0], list):
				for cmd_, arg_ in zip(cmd[0], cmd[1]):
					cmd_(*arg_)
			else:
				print("command: %s" % (cmd[2],))
				cmd[0](*cmd[1])
		else:
			print("key bindings: ")
			print("  Esc: Quit")
			print("  ?: Help")
			for key, val in sorted(self.bindings.items(),
								   key=lambda x: x[1][2]):
				print("  %s: %s" % (key, val[2]))

	def start(self):
		plt.tight_layout()
		plt.show()

	def incr_param(self, param):
		self.params[param][0] += 1
		if self.params[param][0] > self.params[param][2]:
			self.params[param][0] = self.params[param][1]

		self.pretty_print('Param %s is now set to %d' % (param, self.params[param][0]))

	@staticmethod
	def pretty_print(text):
		print(colored("#" * 60, 'green'), "\n",  colored(text, 'green'),"\n", colored("#" * 60, 'green'))

	def move_event(self, event):
		pass

	def click_event(self, event):
		pass

	def release_event(self, event):
		pass

	def scroll_event(self, event):
		pass

	def set_events(self):
		"""
		Register callbacks
		"""

		self.fig.canvas.mpl_connect('key_press_event', self.key_event)
		self.fig.canvas.mpl_connect('motion_notify_event', self.move_event)
		self.fig.canvas.mpl_connect('button_press_event', self.click_event)
		self.fig.canvas.mpl_connect('button_release_event', self.release_event)
		self.fig.canvas.mpl_connect('scroll_event', self.scroll_event)

		self.timer = self.fig.canvas.new_timer(interval=self.simulation_T)
		self.timer.add_callback(self.timer_event, self.ax_dx)

		self.plot_timer = self.fig.canvas.new_timer(interval=50)
		self.plot_timer.add_callback(self.plot_timer_event, self.ax_dx)

