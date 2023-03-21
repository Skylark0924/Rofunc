import sys
import numpy as np
import pandas as pd
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def plot_objects(csv_path: str, objs: dict, meta: dict,
                 show_markers=True, save_gif=False, normalize=False):
    print('[plot_objects] Loading data...')
    print('[plot_objects] data path: ', osp.join(csv_path, f"{meta['Take Name']}.csv"))
    data = pd.read_csv(osp.join(csv_path, f"{meta['Take Name']}.csv"), skiprows=6)
    xlim = (-1200, 1200)
    ylim = (-0.5, 2000)
    zlim = (-1200, 1200)

    if normalize == 'max_divide':
        # Get abosuluete max value
        data /= data.max().max()

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
        ax.set_title(f'Optitrack data visualization frame {data.iloc[i][0]}, time {data.iloc[i][1]}')
        for n, obj in enumerate(objs):
            ptrs = objs[obj]['pose']['Position']
            d = data.iloc[i]
            pt = np.array((d[ptrs['X']], d[ptrs['Y']], d[ptrs['Z']]))
            if not np.isnan(pt).any():
                ax.scatter(pt[0], pt[1], pt[2], marker="o", color=COLORS[n % len(COLORS)], label=obj)
                if show_markers:
                    for marker in objs[obj]['markers']:
                        ptrs = objs[obj]['markers'][marker]['pose']['Position']
                        pt = np.array((d[ptrs['X']], d[ptrs['Y']], d[ptrs['Z']]))
                        if not np.isnan(pt).any():
                                ax.scatter(pt[0], pt[1], pt[2], marker=f"${marker}$", color=COLORS[n % len(COLORS)])
        ax.legend()

    ax.set_xlim(*xlim)  # Because the canvas is cleared, the range of the coordinate axis needs to be reset
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ani = animation.FuncAnimation(fig, update, np.arange(0, dim, 10), interval=100, blit=False)
    if save_gif:
        print('Saving animation...')
        gif_path = osp.join(csv_path, 'optitrack.gif')
        ani.save(gif_path, writer='pillow', fps=10)
        print('{} gif got!'.format(gif_path))
    else:
        print('Showing animation...')
        plt.show()
