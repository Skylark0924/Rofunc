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
        fig.canvas.mpl_connect('button_press_event', self._toggle_pause)

    def _toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.anim.resume()
        else:
            self.anim.pause()
        self.paused = not self.paused

def plot_objects(csv_path: str, objs: dict, meta: dict,
                 show_markers=True, save_gif=False, scale=''):
    """Plots the objects in the objs dict.
    The default bounding box for the plot is:
    xlim = (-1200, 1200)
    ylim = (-0.5, 2000)
    zlim = (-1200, 1200)

    For more precise plots, we offer two scaling options:
    '' -  No scaling, use default bounding box
    'max_scale" - scales the data tp a (-1, 1). Respects original offset to center of scene and aspect ratio.
    'center_scale" - scales the plot to a centered (-1, 1) box. Respects the original aspect ratio.


    Args:
        csv_path (str): Path to the csv file.
        objs (dict): Dictionary of objects to plot.
        meta (dict): Dictionary of metadata.
        show_markers (bool, optional): Whether to show markers. Defaults to True.
        save_gif (bool, optional): Whether to save the gif. Defaults to False.
        scale (str, optional): Normalization method. Defaults to ''.
    """
    print('[plot_objects] Loading data...')
    print('[plot_objects] data path: ', osp.join(csv_path, f"{meta['Take Name']}.csv"))
    data = pd.read_csv(osp.join(csv_path, f"{meta['Take Name']}.csv"), skiprows=6)
    xlim = (-1200, 1200)
    ylim = (-0.5, 2000)
    zlim = (-1200, 1200)
    objs_loc = copy.deepcopy(objs)

    t, s = 0, 1

    mean = 0
    for obj in objs_loc:
        ptrs = objs_loc[obj]['pose']['Position']
        objs_loc[obj]['pose']['Position']['data'] = data.iloc[:, [ptrs['X'], ptrs['Y'], ptrs['Z']]].to_numpy()
        pt = objs_loc[obj]['pose']['Position']['data']
        mean += np.mean(pt, axis=0)
        if show_markers:
            for marker in objs_loc[obj]['markers']:
                ptrs = objs_loc[obj]['markers'][marker]['pose']['Position']
                objs_loc[obj]['markers'][marker]['data'] = data.iloc[:, [ptrs['X'], ptrs['Y'], ptrs['Z']]].to_numpy()
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
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f'Optitrack data visualization frame {int(data.iloc[i][0]):05}, time {data.iloc[i][1]:07.3f}')
        for n, obj in enumerate(objs_loc):
            pts = objs_loc[obj]['pose']['Position']['data']
            pts = (pts - t) * s
            if not np.isnan(pts[i]).any():
                ax.scatter(*pts[i], marker="o", color=COLORS[n % len(COLORS)], label=obj)
                if show_markers:
                    for marker in objs_loc[obj]['markers']:
                        pts = objs_loc[obj]['markers'][marker]['data']
                        pts = (pts - t) * s
                        if not np.isnan(pt[i]).any():
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
