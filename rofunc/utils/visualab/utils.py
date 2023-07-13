"""
 Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import artist
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3D, Text3D

import rofunc as rf
from rofunc.utils.robolab.coord.transform import check_rot_matrix


def set_axis(ax, data=None, labels=None, elev=45, azim=45, roll=0):
    """
    Set the axis of the figure.
    :param ax:
    :param data: [X, Y, Z]
    :param labels:
    :param elev:
    :param azim:
    :param roll:
    :return:
    """
    ax.view_init(elev=elev, azim=azim, roll=roll)
    # ax.set_aspect('equal', 'box')
    ax.set_box_aspect([1, 1, 1])
    if labels is None:
        labels = ['x', 'y', 'z']
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    if data is not None:
        X, Y, Z = data
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 1.2
        mid_x = (X.max() + X.min()) * 0.5
        mid_y = (Y.max() + Y.min()) * 0.5
        mid_z = (Z.max() + Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


def save_img(fig, save_dir, fig_name=None, dpi=300, transparent=False, format=None):
    if format is None:
        format = ['eps', 'png']
    rf.utils.create_dir(save_dir)
    if fig_name is None:
        nb_files = len(os.listdir(save_dir))
        fig_name = 'fig_{}'.format(nb_files)
    for f in format:
        full_fig_name = '{}.{}'.format(fig_name, f)
        save_path = os.path.join(save_dir, full_fig_name)
        fig.savefig(save_path, dpi=dpi, transparent=transparent, format=f)


class Frame(artist.Artist):
    """A Matplotlib artist that displays a frame represented by its basis.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    label : str, optional (default: None)
        Name of the frame

    s : float, optional (default: 1)
        Length of basis vectors

    draw_label_indicator : bool, optional (default: True)
        Controls whether the line from the frame origin to frame label is
        drawn.

    Other arguments except 'c' and 'color' are passed on to Line3D.
    """

    def __init__(self, A2B, label=None, s=1.0, **kwargs):
        super(Frame, self).__init__()

        if "c" in kwargs:
            kwargs.pop("c")
        if "color" in kwargs:
            kwargs.pop("color")

        self.draw_label_indicator = kwargs.pop("draw_label_indicator", True)

        self.s = s

        self.x_axis = Line3D([], [], [], color="r", **kwargs)
        self.y_axis = Line3D([], [], [], color="g", **kwargs)
        self.z_axis = Line3D([], [], [], color="b", **kwargs)

        self.draw_label = label is not None
        self.label = label

        if self.draw_label:
            if self.draw_label_indicator:
                self.label_indicator = Line3D([], [], [], color="k", **kwargs)
            self.label_text = Text3D(0, 0, 0, text="", zdir="x")

        self.set_data(A2B, label)

    def set_data(self, A2B, label=None):
        """Set the transformation data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Transform from frame A to frame B

        label : str, optional (default: None)
            Name of the frame
        """
        R = A2B[:3, :3]
        p = A2B[:3, 3]

        for d, b in enumerate([self.x_axis, self.y_axis, self.z_axis]):
            b.set_data(np.array([p[0], p[0] + self.s * R[0, d]]),
                       np.array([p[1], p[1] + self.s * R[1, d]]))
            b.set_3d_properties(np.array([p[2], p[2] + self.s * R[2, d]]))

        if self.draw_label:
            if label is None:
                label = self.label
            label_pos = p + 0.5 * self.s * (R[:, 0] + R[:, 1] + R[:, 2])

            if self.draw_label_indicator:
                self.label_indicator.set_data(
                    np.array([p[0], label_pos[0]]),
                    np.array([p[1], label_pos[1]]))
                self.label_indicator.set_3d_properties(
                    np.array([p[2], label_pos[2]]))

            self.label_text.set_text(label)
            self.label_text.set_position([label_pos[0], label_pos[1]])
            self.label_text.set_3d_properties(label_pos[2], zdir="x")

    @artist.allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        """Draw the artist."""
        for b in [self.x_axis, self.y_axis, self.z_axis]:
            b.draw(renderer, *args, **kwargs)
        if self.draw_label:
            if self.draw_label_indicator:
                self.label_indicator.draw(renderer, *args, **kwargs)
            self.label_text.draw(renderer, *args, **kwargs)
        super(Frame, self).draw(renderer, *args, **kwargs)

    def add_frame(self, axis):
        """Add the frame to a 3D axis."""
        for b in [self.x_axis, self.y_axis, self.z_axis]:
            axis.add_line(b)
        if self.draw_label:
            if self.draw_label_indicator:
                axis.add_line(self.label_indicator)
            axis._add_text(self.label_text)


def make_3d_axis(ax_s, pos=111, unit=None, n_ticks=5):
    """Generate new 3D axis.

    Parameters
    ----------
    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    pos : int, optional (default: 111)
        Position indicator (nrows, ncols, plot_number)

    unit : str, optional (default: None)
        Unit of axes. For example, 'm', 'cm', 'km', ...
        The unit will be shown in the axis label, for example,
        as 'X [m]'.

    n_ticks : int, optional (default: 5)
        Number of ticks on each axis

    Returns
    -------
    ax : Matplotlib 3d axis
        New axis
    """
    try:
        ax = plt.subplot(pos, projection="3d", aspect="equal")
    except NotImplementedError:
        # HACK: workaround for bug in new matplotlib versions (ca. 3.02):
        # "It is not currently possible to manually set the aspect"
        ax = plt.subplot(pos, projection="3d")

    if unit is None:
        xlabel = "X"
        ylabel = "Y"
        zlabel = "Z"
    else:
        xlabel = "X [%s]" % unit
        ylabel = "Y [%s]" % unit
        zlabel = "Z [%s]" % unit

    plt.setp(
        ax, xlim=(-ax_s, ax_s), ylim=(-ax_s, ax_s), zlim=(-ax_s, ax_s),
        xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

    ax.xaxis.set_major_locator(MaxNLocator(n_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(n_ticks))
    ax.zaxis.set_major_locator(MaxNLocator(n_ticks))

    try:
        ax.xaxis.pane.set_color("white")
        ax.yaxis.pane.set_color("white")
        ax.zaxis.pane.set_color("white")
    except AttributeError:  # pragma: no cover
        # fallback for older versions of matplotlib, deprecated since v3.1
        ax.w_xaxis.pane.set_color("white")
        ax.w_yaxis.pane.set_color("white")
        ax.w_zaxis.pane.set_color("white")

    return ax


def plot_basis(ax=None, R=None, p=np.zeros(3), s=1.0, ax_s=1,
               strict_check=True, **kwargs):
    """Plot basis of a rotation matrix.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    R : array-like, shape (3, 3), optional (default: I)
        Rotation matrix, each column contains a basis vector

    p : array-like, shape (3,), optional (default: [0, 0, 0])
        Offset from the origin

    s : float, optional (default: 1)
        Scaling of the frame that will be drawn

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise, we print a warning.

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    if ax is None:
        ax = make_3d_axis(ax_s)

    if R is None:
        R = np.eye(3)
    R = check_rot_matrix(R, strict_check=strict_check)

    A2B = np.eye(4)
    A2B[:3, :3] = R
    A2B[:3, 3] = p

    frame = Frame(A2B, s=s, **kwargs)
    frame.add_frame(ax)

    return ax
