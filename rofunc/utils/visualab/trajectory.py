import random

import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from matplotlib.pyplot import cm
from pytransform3d.rotations import matrix_from_quaternion, plot_basis

matplotlib_axes_logger.setLevel('ERROR')


def traj_plot2d(data_lst, title=None):
    plt.figure()
    for data in data_lst:
        plt.plot(data[:, 0], data[:, 1])
    plt.show()


def traj_plot3d(data_lst, title=None, g_ax=None, ori=False):
    if g_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', fc='white')
    else:
        ax = g_ax

    c = cm.tab20c(random.random())
    for i in range(len(data_lst)):
        if i == 0 and title is not None:
            # Plot 3d trajectories
            ax.plot(data_lst[i][:, 0], data_lst[i][:, 1], data_lst[i][:, 2], label='{}'.format(title), c=c)

            # Starting points
            ax.scatter(data_lst[i][0, 0], data_lst[i][0, 1], data_lst[i][0, 2], s=20,
                       label='start point of {}'.format(title), c=c)

            # End points
            ax.scatter(data_lst[i][-1, 0], data_lst[i][-1, 1], data_lst[i][-1, 2], marker='x', s=20, c=c,
                       label='end point of {}'.format(title))
        else:
            ax.plot(data_lst[i][:, 0], data_lst[i][:, 1], data_lst[i][:, 2], c=c)
            ax.scatter(data_lst[i][0, 0], data_lst[i][0, 1], data_lst[i][0, 2], s=20, c=c)
            ax.scatter(data_lst[i][-1, 0], data_lst[i][-1, 1], data_lst[i][-1, 2], marker='x', s=20, c=c)

        if ori:
            data_ori = data_lst[i][:, 3:7]
            for t in range(len(data_ori)):
                R = matrix_from_quaternion(data_ori[t])
                p = data_ori[i][t, :3]
                ax = plot_basis(ax=ax, R=R, p=p, s=0.01)
                ax = plot_basis(ax=ax, R=R, p=p, s=0.01)
    ax.legend()
    if g_ax is None:
        plt.show()


def traj_plot(data_lst, title=None, mode=None, ori=False, g_ax=None):
    """

    Args:
        data_lst: list with 2d array or 3d array
        title:
        mode:
        ori:
        g_ax:

    Returns:

    """
    if mode is None:
        mode = '2d' if len(data_lst[0][0]) == 2 else '3d'

    if mode == '2d':
        traj_plot2d(data_lst, title)
    elif mode == '3d':
        traj_plot3d(data_lst, title=title, g_ax=g_ax, ori=ori)
    else:
        raise Exception('Wrong mode, only support 2d and 3d plot.')
