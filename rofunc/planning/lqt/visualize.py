import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from matplotlib.pyplot import cm
from pytransform3d.rotations import matrix_from_quaternion, plot_basis

matplotlib_axes_logger.setLevel('ERROR')


def plot_2d(param, x_hat_l, x_hat_r, idx_slices, tl, via_point_l, via_point_r):
    plt.figure()
    plt.title("2D Trajectory")
    plt.scatter(x_hat_l[0, 0], x_hat_l[0, 1], c='blue', s=100)
    plt.scatter(x_hat_r[0, 0], x_hat_r[0, 1], c='green', s=100)
    for slice_t in idx_slices:
        plt.scatter(param["muQ_l"][slice_t][0], param["muQ_l"][slice_t][1], c='red', s=100)
        plt.scatter(param["muQ_r"][slice_t][0], param["muQ_r"][slice_t][1], c='orange', s=100)
        plt.plot([param["muQ_l"][slice_t][0], param["muQ_r"][slice_t][0]],
                 [param["muQ_l"][slice_t][1], param["muQ_r"][slice_t][1]], linewidth=2, color='black')
    plt.plot(x_hat_l[:, 0], x_hat_l[:, 1], c='blue')
    plt.plot(x_hat_r[:, 0], x_hat_r[:, 1], c='green')
    plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    fig, axs = plt.subplots(3, 1)
    for i, t in enumerate(tl):
        axs[0].scatter(t, param["muQ_l"][idx_slices[i]][0], c='red')
        axs[0].scatter(t, param["muQ_r"][idx_slices[i]][0], c='orange')
    axs[0].plot(x_hat_l[:, 0], c='blue')
    axs[0].plot(x_hat_r[:, 0], c='green')
    axs[0].set_ylabel("$x_1$")
    axs[0].set_xticks([0, param["nbData"]])
    axs[0].set_xticklabels(["0", "T"])

    for i, t in enumerate(tl):
        axs[1].scatter(t, param["muQ_l"][idx_slices[i]][1], c='red')
        axs[1].scatter(t, param["muQ_r"][idx_slices[i]][1], c='orange')
    axs[1].plot(x_hat_l[:, 1], c='blue')
    axs[1].plot(x_hat_r[:, 1], c='green')
    axs[1].set_ylabel("$x_2$")
    axs[1].set_xlabel("$t$")
    axs[1].set_xticks([0, param["nbData"]])
    axs[1].set_xticklabels(["0", "T"])

    dis_lst = []
    for i in range(len(x_hat_l)):
        dis_lst.append(np.sqrt(np.sum(np.square(x_hat_l[i, :2] - x_hat_r[i, :2]))))

    dis_lst = np.array(dis_lst)
    timestep = np.arange(len(dis_lst))
    axs[2].plot(timestep, dis_lst)
    axs[2].set_ylabel("traj_dis")
    axs[2].set_xlabel("$t$")
    axs[2].set_xticks([0, param["nbData"]])
    axs[2].set_xticklabels(["0", "T"])

    dis_lst = []
    via_point_l = np.array(via_point_l)
    via_point_r = np.array(via_point_r)
    for i in range(len(via_point_l)):
        dis_lst.append(np.sqrt(np.sum(np.square(via_point_l[i, :2] - via_point_r[i, :2]))))

    dis_lst = np.array(dis_lst)
    timestep = np.arange(len(dis_lst))
    axs[3].plot(timestep, dis_lst)

    plt.show()


def plot_3d_uni(x_hat, muQ=None, idx_slices=None, ori=False, save=False, save_file_name=None, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d', fc='white')

    if muQ is not None and idx_slices is not None:
        for slice_t in idx_slices:
            ax.scatter(muQ[slice_t][0], muQ[slice_t][1], muQ[slice_t][2], c='red', s=10)

    # Plot 3d trajectories
    ax.plot(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2], c='black')

    # Starting points
    ax.scatter(x_hat[0, 0], x_hat[0, 1], x_hat[0, 2], c='black', s=20)

    if ori:
        ori = x_hat[:, 3:7]
        for t in range(0, len(ori), 10):
            R = matrix_from_quaternion(ori[t])
            p = x_hat[t, :3]
            ax = plot_basis(ax=ax, R=R, p=p, s=0.001)

    if save:
        assert save_file_name is not None
        np.save(save_file_name, np.array(x_hat))
    plt.show()


def plot_3d_bi(x_hat_l, x_hat_r, muQ_l=None, muQ_r=None, idx_slices=None, ori=True, save=False, save_file_name=None,
               ax=None):
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')

    if muQ_l is not None and muQ_r is not None and idx_slices is not None:
        for slice_t in idx_slices:
            ax.scatter(muQ_l[slice_t][0], muQ_l[slice_t][1], muQ_l[slice_t][2], c='red', s=10)
            ax.scatter(muQ_r[slice_t][0], muQ_r[slice_t][1], muQ_r[slice_t][2], c='orange', s=10)

    if not isinstance(x_hat_l, list):
        if len(x_hat_l.shape) == 2:
            x_hat_l = np.expand_dims(x_hat_l, axis=0)
            x_hat_r = np.expand_dims(x_hat_r, axis=0)

    for i in range(len(x_hat_l)):
        c_l = cm.tab20c(random.random())
        c_r = cm.tab20b(random.random())

        # Plot 3d trajectories
        ax.plot(x_hat_l[i][:, 0], x_hat_l[i][:, 1], x_hat_l[i][:, 2], label='left arm', c=c_l)
        ax.plot(x_hat_r[i][:, 0], x_hat_r[i][:, 1], x_hat_r[i][:, 2], label='right arm', c=c_r)

        # Starting points
        ax.scatter(x_hat_l[i][0, 0], x_hat_l[i][0, 1], x_hat_l[i][0, 2], s=20, label='left start point', c=c_l)
        ax.scatter(x_hat_r[i][0, 0], x_hat_r[i][0, 1], x_hat_r[i][0, 2], s=20, label='right start point', c=c_r)

        # End points
        ax.scatter(x_hat_l[i][-1, 0], x_hat_l[i][-1, 1], x_hat_l[i][-1, 2], marker='x', s=20, c=c_l,
                   label='left end point')
        ax.scatter(x_hat_r[i][-1, 0], x_hat_r[i][-1, 1], x_hat_r[i][-1, 2], marker='x', s=20, c=c_r,
                   label='right end point')

        if ori:
            l_ori = x_hat_l[i][:, 3:7]
            r_ori = x_hat_r[i][:, 3:7]
            for t in range(len(l_ori)):
                R_l = matrix_from_quaternion(l_ori[t])
                R_r = matrix_from_quaternion(r_ori[t])
                p_l = x_hat_l[i][t, :3]
                p_r = x_hat_r[i][t, :3]
                ax = plot_basis(ax=ax, R=R_l, p=p_l, s=0.01)
                ax = plot_basis(ax=ax, R=R_r, p=p_r, s=0.01)

    if save:
        assert save_file_name is not None
        np.save(save_file_name[0], np.array(x_hat_l))
        np.save(save_file_name[1], np.array(x_hat_r))
    ax.legend()
    plt.show()
