import sys
import copy
import numpy as np
import pandas as pd
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


class PauseAnimation(object):
    def __init__(self, fig, *args, **kwargs):
        self.anim = animation.FuncAnimation(fig, *args, **kwargs)
        self.paused = False
        fig.canvas.mpl_connect('key_press_event', self._toggle_pause)

    def _toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.anim.resume()
        else:
            self.anim.pause()
        self.paused = not self.paused


def visualize_objects(parent_dir: str, objs: dict, meta: dict, show_markers: bool = True, save_gif: bool = False,
                      scale: str = '', up_axis: str = 'Y'):
    """
    Plots the objects in the objs dict.
    The default bounding box for the plot is:
    xlim = (-1200, 1200)
    ylim = (-0.5, 2000)
    zlim = (-1200, 1200)

    For more precise plots, we offer two scaling options:
    '' -  No scaling, use default bounding box
    'max_scale' - scales the data tp a (-1, 1). Respects original offset to center of scene and aspect ratio.
    'center_scale' - scales the plot to a centered (-1, 1) box. Respects the original aspect ratio.
    :param parent_dir: Directory where the data is stored.
    :param objs: Dictionary of objects to plot.
    :param meta: Dictionary of metadata.
    :param show_markers:
    :param save_gif: Whether to save the gif. Defaults to False.
    :param scale: Normalization method. Defaults to ''.
    :param up_axis: Axis to be considered as up. Defaults to 'Y'.
    :return:
    """
    data = pd.read_csv(osp.join(parent_dir, f"{meta['Take Name']}.csv"), skiprows=6)
    xlim = (-1200, 1200)
    ylim = (-0.5, 2000)
    zlim = (-1200, 1200)
    objs_loc = copy.deepcopy(objs)
    up_axis = up_axis.upper()

    t, s = 0, 1

    mean = 0
    for obj in objs_loc:
        ptrs = objs_loc[obj]['pose']['Position']
        if up_axis == 'Y':
            objs_loc[obj]['pose']['Position']['data'] = data.iloc[:, [ptrs['X'], ptrs['Z'], ptrs['Y']]].to_numpy()
        elif up_axis == 'Z':
            objs_loc[obj]['pose']['Position']['data'] = data.iloc[:, [ptrs['X'], ptrs['Y'], ptrs['Z']]].to_numpy()
        else:
            raise ValueError(f'Invalid up_axis: {up_axis}')
        pt = objs_loc[obj]['pose']['Position']['data']
        mean += np.mean(pt, axis=0)
        if show_markers:
            for marker in objs_loc[obj]['markers']:
                ptrs = objs_loc[obj]['markers'][marker]['pose']['Position']
                if up_axis == 'Y':
                    objs_loc[obj]['markers'][marker]['data'] = data.iloc[:,
                                                               [ptrs['X'], ptrs['Z'], ptrs['Y']]].to_numpy()
                elif up_axis == 'Z':
                    objs_loc[obj]['markers'][marker]['data'] = data.iloc[:,
                                                               [ptrs['X'], ptrs['Y'], ptrs['Z']]].to_numpy()
                else:
                    raise ValueError(f'Invalid up_axis: {up_axis}')
    mean /= len(objs_loc)

    if scale == 'max_scale':
        s = 1 / data.iloc[:, 2:].abs().max().max()
        xlim = (-1, 1)
        ylim = (-1, 1)
        zlim = (-1, 1)
    elif scale == 'center_scale':
        max = 0
        for obj in objs_loc:
            d = objs_loc[obj]['pose']['Position']['data']
            d -= mean
            max = np.max(np.abs(d)) if np.max(np.abs(d)) > max else max
            objs_loc[obj]['pose']['Position']['data'] = d
            for marker in objs_loc[obj]['markers']:
                objs_loc[obj]['markers'][marker]['data'] -= mean
        s = 1 / (0.8 * max)
        xlim = (-1, 1)
        ylim = (-1, 1)
        zlim = (-1, 1)

    fig = plt.figure(num=f'Optitrack data visualization - {meta["Take Name"]}')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(*xlim)  # Because the canvas is cleared, the range of the coordinate axis needs to be reset
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    dim = len(data)

    def update(i):
        ax.cla()
        ax.set_xlim(*xlim)  # Because the canvas is cleared, the range of the coordinate axis needs to be reset
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel('x')
        if up_axis == 'Y':
            ax.set_ylabel('z')
            ax.set_zlabel('y')
        elif up_axis == 'Z':
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        else:
            raise ValueError(f'Invalid up_axis: {up_axis}')
        ax.set_title(f'Optitrack data visualization frame {int(data.iloc[i][0]):05}, time {data.iloc[i][1]:07.3f}')
        for n, obj in enumerate(objs_loc):
            pts = objs_loc[obj]['pose']['Position']['data']
            pts = (pts - t) * s
            if not np.isnan(pts[i]).any():
                ax.scatter(*pts[i], marker="o", color=COLORS[n % len(COLORS)], label=obj)
                ax.text(*pts[i], obj[-5:])
                if show_markers:
                    for marker in objs_loc[obj]['markers']:
                        pts = objs_loc[obj]['markers'][marker]['data']
                        pts = (pts - t) * s
                        if not np.isnan(pts[i]).any():
                            ax.scatter(*pts[i], marker=f"${marker}$", color=COLORS[n % len(COLORS)])

        ax.legend()

    ax.set_xlim(*xlim)  # Because the canvas is cleared, the range of the coordinate axis needs to be reset
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    # ani = animation.FuncAnimation(fig, update, np.arange(0, dim, 10), interval=100, blit=False)
    ani = PauseAnimation(fig, update, np.arange(0, dim, 10), interval=100, blit=False)
    if save_gif:
        print('Saving animation...')
        gif_path = osp.join(csv_path, 'optitrack.gif')
        ani.save(gif_path, writer='pillow', fps=10)
        print('{} gif got!'.format(gif_path))
    else:
        print('Showing animation...')
        plt.show()
