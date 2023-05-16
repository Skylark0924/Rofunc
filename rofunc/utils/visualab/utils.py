import os
import rofunc as rf
import numpy as np


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
