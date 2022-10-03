from .demos import *
from ..utils import angle_to_rotation


class CoordinateSys2D(object):
    x, alpha, d = np.array([0, 0]), 0, 0

    _angle_based = True
    _A = None
    _size = 10

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value

    @property
    def A(self):
        if self._A is None:
            self._A = angle_to_rotation(self.alpha).T

        return self._A

    @A.setter
    def A(self, A):
        self._angle_based = False
        self._A = A

    def get_points(self, size=None, shape='L'):

        if size is not None: self.size = size
        mat = self.size * self.A.T

        if shape == 'L':
            return np.vstack([self.x + mat[0], self.x, self.x + mat[1]])

        elif shape == 'T':
            return np.vstack([self.x - mat[0], self.x, self.x + mat[0], self.x, self.x + mat[1] / 4.])

    def wall_reaction_force(self, x, dx, size=None, k=5e4, thickness=3., sensor_mode=0):
        n = self.A[:, 1]  # normal vector of the wall
        p = self.A[:, 0]  # perpendicular
        kv = 2. * k ** 0.5
        kv_tang = kv / 2.
        if size is not None: self.size = size

        sf = np.zeros(1)

        if sensor_mode == 1:
            sf = np.minimum(np.abs(n.dot(x - self.x)), 40.)

        if (np.abs(p.dot(x - self.x)) < self.size) and (np.abs(n.dot(x - self.x)) < thickness):
            f = -k * n * np.maximum(n.dot(x - self.x), 0)

            if sensor_mode == 0:
                sf += -k * np.maximum(n.dot(x - self.x), 0)

            if n.dot(x - self.x) > 0:
                f += -kv * dx  # * np.maximum(n.dot(dx)/dx, 0.) #- kv_tang * p * p.dot(dx)
            # sf += np.norm(-kv * np.maximum(n.dot(dx) / dx, 0.))

        else:
            f = np.zeros(2)

        return f, sf


class MultiCsInteractive(object):
    def __init__(self, nb_experts=2, **kwargs):
        self.nb_experts = nb_experts
        self.expert_cs = [CoordinateSys2D() for i in range(self.nb_experts)]
        self.object_experts_cs = CoordinateSys2D()
        self.curr_cs = {'A': [], 'b': []}

    def add_bindings(self):
        for i in range(1, self.nb_experts + 1):
            self.bindings['%d' % (i)] = (self.select_cs, [i], "select frame %d" % i)

    def set_plots(self):
        for i in range(self.nb_experts):
            self.plots['cs_%d' % i] = self.ax_x.plot([], [], lw=2)[0]

        self.plots['cs_obj'] = self.ax_x.plot([], [], lw=2)[0]

    def scroll_event(self, event):
        if event.key in [str(i + 1) for i in range(self.nb_experts)]:
            exp_idx = int(event.key) - 1

            incr = np.pi / 16
            if event.button == 'up':
                self.expert_cs[exp_idx].alpha += incr
            elif event.button == 'down':
                self.expert_cs[exp_idx].alpha -= incr
            self.expert_cs[exp_idx].A = None
            self.update_cs(exp_idx)

            self.fig.canvas.draw()

    def select_cs(self, idx):
        self.pretty_print('Coordinate system %d selected' % idx)

    def update_cs(self, exp_idx, obj_exp=False):
        if obj_exp:
            pts = self.object_experts_cs.get_points()
            self.plots['cs_obj'].set_data(pts[:, 0], pts[:, 1])
            return
        pts = self.expert_cs[exp_idx].get_points()
        self.plots['cs_%d' % exp_idx].set_data(pts[:, 0], pts[:, 1])

    def move_event(self, event):
        cs_updated = False

        if event.key in [str(i + 1) for i in range(self.nb_experts)]:
            cs_updated = True
            exp_idx = int(event.key) - 1

            self.expert_cs[exp_idx].x = np.copy(self.curr_mouse_pos)
            self.update_cs(exp_idx)

        return cs_updated


class MutliCsInteractiveDemos(InteractiveDemos, MultiCsInteractive):
    """
	GUI for recording demonstrations in 2D with moveable frame of reference
	"""

    def __init__(self, nb_experts=2, erase=False, **kwargs):
        MultiCsInteractive.__init__(self, nb_experts=nb_experts, **kwargs)
        InteractiveDemos.__init__(self, **kwargs)

        if erase:
            self.clear_demos()

        if 'A' not in self.demos:
            for s in ['A', 'b', 'obj_x']:  # adding list for coordinate systems
                self.demos[s] = []

        MultiCsInteractive.add_bindings(self)

    def scroll_event(self, event):
        MultiCsInteractive.scroll_event(self, event)

    def timer_event(self, event):
        super(MutliCsInteractiveDemos, self).timer_event(event)

        self.curr_cs['A'] += [np.copy(np.array([exp.A for exp in self.expert_cs]))]
        self.curr_cs['b'] += [np.copy(np.array([exp.x for exp in self.expert_cs]))]
        self.curr_demo_obj += [np.copy(self.object_experts_cs.x)]

    def set_plots(self):

        InteractiveDemos.set_plots(self)
        MultiCsInteractive.set_plots(self)

    def finish_demo(self):
        for s in ['A', 'b']:
            self.demos[s] += [np.copy(np.array(self.curr_cs[s]))]
            self.curr_cs[s] = []

        self.demos['obj_x'] += [np.copy(np.array(self.curr_demo_obj))]
        self.curr_demo_obj = []

        super(MutliCsInteractiveDemos, self).finish_demo()

    def move_event(self, event):
        super(MutliCsInteractiveDemos, self).move_event(event)
        MultiCsInteractive.move_event(self, event)
        #
        if event.key in [str(i + 1) for i in range(self.nb_experts)]:
            exp_idx = int(event.key) - 1
            self.expert_cs[exp_idx].x = np.copy(self.curr_mouse_pos)
            self.update_cs(exp_idx)
            self.fig.canvas.draw()

    # if event.key == 'w':
    # 	self.object_experts_cs.x = self.curr_mouse_pos
    # 	self.object_experts_cs.d = self.object_experts_cs.x - self.robot_pos
    # 	self.update_cs(0, obj_exp=True)
    # 	self.fig.canvas.draw()
    # else:
    # 	self.object_experts_cs.x = self.object_experts_cs.d + self.robot_pos
    # 	self.update_cs(0, obj_exp=True)
    # 	self.fig.canvas.draw()
