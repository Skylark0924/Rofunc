import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def plot_skeleton(skeleton_data_path: str, save_gif=False):
    if not os.path.isdir(skeleton_data_path):
        raise Exception('The skeleton_data_path must be a folder with multiple .npy files about the skeleton. '
                        'Please use get_skeleton to generate the complete skeleton data.')

    print('{} animate start!'.format(skeleton_data_path.split('/')[-1]))
    labels = os.listdir(skeleton_data_path)
    data_dict = {}
    min_x, min_y, min_z = np.inf, np.inf, np.inf
    max_x, max_y, max_z = -np.inf, -np.inf, -np.inf
    for label in labels:
        if label.endswith('npy'):
            np_data = np.load(os.path.join(skeleton_data_path, "{}".format(label)))
            min_x = min(min_x, np_data[:, 0].min())
            min_y = min(min_y, np_data[:, 1].min())
            min_z = min(min_z, np_data[:, 2].min())
            max_x = max(max_x, np_data[:, 0].max())
            max_y = max(max_y, np_data[:, 1].max())
            max_z = max(max_z, np_data[:, 2].max())
            if len(np_data.shape) == 1:
                continue
            data_dict['{}'.format(label.split('.')[0].split('_')[-1])] = np_data

    max_range = np.array([max_x - min_x, max_y - min_y, max_z - min_z]).max() / 3
    dim = len(data_dict['Head'])

    fig = plt.figure()
    # ax = Axes3D(fig, fc='white')
    ax = plt.axes(projection='3d')
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlim(min_x - max_range, max_x + max_range)
    ax.set_ylim(min_y - max_range, max_y + max_range)
    ax.set_zlim(min_z - max_range, max_z + max_range)

    def update(index):
        ax.cla()  # Clear the canvas
        ax.set_xlim(min_x - max_range, max_x + max_range)
        ax.set_ylim(min_y - max_range, max_y + max_range)
        ax.set_zlim(min_z - max_range, max_z + max_range)
        ax.text2D(0.05, 0.95, "Frame: {}".format(index), transform=ax.transAxes)
        for label in labels:
            if label.split('.')[-1] == 'npy':
                if 'finger' not in label:
                    x, y, z = data_dict['{}'.format(label.split('.')[0].split('_')[-1])][index, :3]
                    ax.scatter(x, y, z, c='r')

    ax.set_xlim(min_x - max_range, max_x + max_range)
    ax.set_ylim(min_y - max_range, max_y + max_range)
    ax.set_zlim(min_z - max_range, max_z + max_range)
    ani = animation.FuncAnimation(fig, update, np.arange(0, dim, 10), interval=100, blit=False)
    if save_gif:
        ani.save(os.path.join(skeleton_data_path, '{}.gif'.format(skeleton_data_path.split('/')[-1])),
                 writer='pillow', fps=10)
        print('{} gif got!'.format(skeleton_data_path.split('/')[-1]))
    else:
        plt.show()


def plot_skeleton_batch(skeleton_dir, save_gif=True):
    skeletons = os.listdir(skeleton_dir)
    for skeleton in tqdm(skeletons):
        # if "chenzui" not in skeleton:
        #     continue
        skeleton_path = os.path.join(skeleton_dir, skeleton, 'segment')
        if os.path.isdir(skeleton_path):
            plot_skeleton(skeleton_path, save_gif)
