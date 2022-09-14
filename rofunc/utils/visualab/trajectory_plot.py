import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def traj_plot2d(data_lst, title=None):
    plt.figure()
    for data in data_lst:
        plt.plot(data[:, 0], data[:, 1])
    plt.show()


def traj_plot3d(data_lst, title=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for data in data_lst:
        ax.plot(data[:, 0], data[:, 1], data[:, 2])
    plt.show()


def traj_plot(data_lst, title=None, mode=None):
    """

    Args:
        data_lst: list with 2d array or 3d array
        title:
        mode:

    Returns:

    """
    if mode is None:
        mode = '2d' if len(data_lst[0][0]) == 2 else '3d'

    if mode == '2d':
        traj_plot2d(data_lst, title)
    elif mode == '3d':
        traj_plot3d(data_lst, title)
    else:
        raise Exception('Wrong mode, only support 2d and 3d plot.')
