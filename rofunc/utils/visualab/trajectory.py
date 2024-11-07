#  Copyright (C) 2024, Junjia Liu
#
#  This file is part of Rofunc.
#
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
#
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
#
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com

import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from matplotlib.pyplot import cm

from rofunc.utils.robolab.coord.transform import homo_matrix_from_quaternion
from rofunc.utils.visualab.utils import set_axis, plot_basis

matplotlib_axes_logger.setLevel('ERROR')


def traj_plot2d(data_lst: List, legend: str = None, title: str = None, g_ax=None):
    """
    Plot multiple 2d trajectories

    Example::

        >>> import rofunc as rf
        >>> import numpy as np
        >>> data_lst = [np.array([[0, 0], [1, 1], [2, 3]]),
        ...             np.array([[0, 0], [1, 2], [4, 2]])]
        >>> fig = rf.visualab.traj_plot2d(data_lst, legend='test')
        >>> plt.show()

    :param data_lst:
    :param legend:
    :param title:
    :param g_ax:
    :return:
    """
    if g_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)  # , fc='white'
    else:
        ax = g_ax
    for i in range(len(data_lst)):
        if i == 0 and legend is not None:
            ax.plot(data_lst[i][:, 0], data_lst[i][:, 1], label='{}'.format(legend))

            # Starting points
            ax.scatter(data_lst[i][0, 0], data_lst[i][0, 1], s=20, label='start point of {}'.format(legend))

            # End points
            ax.scatter(data_lst[i][-1, 0], data_lst[i][-1, 1], marker='x', s=20, label='end point of {}'.format(legend))
            ax.legend()
        else:
            ax.plot(data_lst[i][:, 0], data_lst[i][:, 1])
            ax.scatter(data_lst[i][0, 0], data_lst[i][0, 1], s=20)
            ax.scatter(data_lst[i][-1, 0], data_lst[i][-1, 1], s=20)

    if title is not None:
        ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    if g_ax is None:
        return fig


def traj_plot3d(data_lst: List, legend: str = None, title: str = None, g_ax=None, ori: bool = False):
    """
    Plot multiple 3d trajectories

    Example::

        >>> import rofunc as rf
        >>> import numpy as np
        >>> data_lst = [np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        ...             np.array([[1, 0, 0], [1, 4, 6], [2, 4, 3]])]
        >>> fig = rf.visualab.traj_plot3d(data_lst, legend='test')
        >>> plt.show()

    :param data_lst: the list of trajectories
    :param legend: the legend of the figure
    :param title: the title of the figure
    :param g_ax: whether to plot on a global axis
    :param ori: plot orientation or not
    :return: the figure
    """
    if g_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', fc='white')
    else:
        ax = g_ax

    c = cm.tab20c(random.random())
    for i in range(len(data_lst)):
        if i == 0 and legend is not None:
            # Plot 3d trajectories
            ax.plot(data_lst[i][:, 0], data_lst[i][:, 1], data_lst[i][:, 2], label='{}'.format(legend), c=c)

            # Starting points
            ax.scatter(data_lst[i][0, 0], data_lst[i][0, 1], data_lst[i][0, 2], s=20,
                       label='start point of {}'.format(legend), c=c)

            # End points
            ax.scatter(data_lst[i][-1, 0], data_lst[i][-1, 1], data_lst[i][-1, 2], marker='x', s=20, c=c,
                       label='end point of {}'.format(legend))
            ax.legend()
        else:
            ax.plot(data_lst[i][:, 0], data_lst[i][:, 1], data_lst[i][:, 2], c=c)
            ax.scatter(data_lst[i][0, 0], data_lst[i][0, 1], data_lst[i][0, 2], s=20, c=c)
            ax.scatter(data_lst[i][-1, 0], data_lst[i][-1, 1], data_lst[i][-1, 2], marker='x', s=20, c=c)

        if ori and len(data_lst[i][0]) >= 7:
            data_ori = data_lst[i][:, 3:7]
            for t in range(len(data_ori)):
                try:
                    R = homo_matrix_from_quaternion(data_ori[t])
                except:
                    pass
                p = data_lst[i][t, :3]
                if t % 20 == 0:
                    ax = plot_basis(ax=ax, R=R, p=p, s=1)
                    ax = plot_basis(ax=ax, R=R, p=p, s=1)
    if title is not None:
        ax.set_title(title, fontsize=12, fontweight='bold')

    # # Create cubic bounding box to simulate equal aspect ratio
    # X_max = max([data_lst[i][:, 0].max() for i in range(len(data_lst))])
    # X_min = min([data_lst[i][:, 0].min() for i in range(len(data_lst))])
    # Y_max = max([data_lst[i][:, 1].max() for i in range(len(data_lst))])
    # Y_min = min([data_lst[i][:, 1].min() for i in range(len(data_lst))])
    # Z_max = max([data_lst[i][:, 2].max() for i in range(len(data_lst))])
    # Z_min = min([data_lst[i][:, 2].min() for i in range(len(data_lst))])
    # max_range = np.array([X_max - X_min, Y_max - Y_min, Z_max - Z_min]).max()
    # Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X_max + X_min)
    # Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y_max + Y_min)
    # Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z_max + Z_min)
    # # Comment or uncomment following both lines to test the fake bounding box:
    # for xb, yb, zb in zip(Xb, Yb, Zb):
    #     ax.plot([xb], [yb], [zb], 'w')

    tmp_data = np.vstack((data_lst[i] for i in range(len(data_lst))))
    set_axis(ax, data=[tmp_data[:, 0], tmp_data[:, 1], tmp_data[:, 2]])
    plt.tight_layout()
    if g_ax is None:
        return fig


def traj_plot(data_lst: List, legend: str = None, title: str = None, mode: str = None, ori: bool = False, g_ax=None):
    """
    Plot 2d or 3d trajectories

    Example::

        >>> import rofunc as rf
        >>> import numpy as np
        >>> data_lst = [np.array([[0, 0], [1, 1], [2, 3]]),
        ...             np.array([[0, 0], [1, 2], [4, 2]])]
        >>> fig = rf.visualab.traj_plot(data_lst, legend='test')
        >>> plt.show()

        >>> data_lst = [np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        ...             np.array([[1, 0, 0], [1, 4, 6], [2, 4, 3]])]
        >>> fig = rf.visualab.traj_plot(data_lst, legend='test')
        >>> plt.show()

    :param data_lst: list with 2d array or 3d array
    :param legend: legend of the figure
    :param title: title of the figure
    :param mode: '2d' or '3d'
    :param ori: plot orientation or not
    :param g_ax: global axis
    :return:
    """
    if mode is None:
        mode = '2d' if len(data_lst[0][0]) == 2 else '3d'
    if mode == '2d':
        fig = traj_plot2d(data_lst, legend, title, g_ax)
    elif mode == '3d':
        fig = traj_plot3d(data_lst, legend, title, g_ax, ori)
    else:
        raise Exception('Wrong mode, only support 2d and 3d plot.')
    return fig
