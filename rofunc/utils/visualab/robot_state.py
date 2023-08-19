"""
Warning: This file is deprecated and will be removed in a future release.
"""

import matplotlib.pyplot as plt
import numpy as np


def state_plot(pos: np.ndarray, vel: np.ndarray, effort: np.ndarray, joint_index, g_ax=None):
    if g_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, fc='white')
    else:
        ax = g_ax

    if pos is not None:
        ax.plot(pos[:, joint_index], label='pos')
    if vel is not None:
        ax.plot(vel[:, joint_index], label='vel')
    if effort is not None:
        ax.plot(effort[:, joint_index], label='effort')

    ax.legend()
    ax.set_title('State', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(-5, 5)
    plt.tight_layout()
    if g_ax is None:
        plt.show()


if __name__ == '__main__':
    state = np.load(
        '/examples/simulator/state7.npy',
        allow_pickle=True)[10: 310]
    pos = state[:, :, 0]
    vel = state[:, :, 1]
    effort = state[:, :, 2]
    # for i in range(7, 13):
    #     state_plot(pos, vel, effort, i)
    state_plot(pos, vel, effort, 13)
