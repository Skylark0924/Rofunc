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

import matplotlib.pyplot as plt
import numpy as np
import rofunc as rf


# def plot_2d(cfg, x_hat_l, x_hat_r, idx_slices, tl, via_point_l, via_point_r):
#     # TODO: check
#     plt.figure()
#     plt.title("2D Trajectory")
#     plt.scatter(x_hat_l[0, 0], x_hat_l[0, 1], c='blue', s=100)
#     plt.scatter(x_hat_r[0, 0], x_hat_r[0, 1], c='green', s=100)
#     for slice_t in idx_slices:
#         plt.scatter(param["muQ_l"][slice_t][0], param["muQ_l"][slice_t][1], c='red', s=100)
#         plt.scatter(param["muQ_r"][slice_t][0], param["muQ_r"][slice_t][1], c='orange', s=100)
#         plt.plot([param["muQ_l"][slice_t][0], param["muQ_r"][slice_t][0]],
#                  [param["muQ_l"][slice_t][1], param["muQ_r"][slice_t][1]], linewidth=2, color='black')
#     plt.plot(x_hat_l[:, 0], x_hat_l[:, 1], c='blue')
#     plt.plot(x_hat_r[:, 0], x_hat_r[:, 1], c='green')
#     plt.axis("off")
#     plt.gca().set_aspect('equal', adjustable='box')
#
#     fig, axs = plt.subplots(3, 1)
#     for i, t in enumerate(tl):
#         axs[0].scatter(t, param["muQ_l"][idx_slices[i]][0], c='red')
#         axs[0].scatter(t, param["muQ_r"][idx_slices[i]][0], c='orange')
#     axs[0].plot(x_hat_l[:, 0], c='blue')
#     axs[0].plot(x_hat_r[:, 0], c='green')
#     axs[0].set_ylabel("$x_1$")
#     axs[0].set_xticks([0, cfg.nbData])
#     axs[0].set_xticklabels(["0", "T"])
#
#     for i, t in enumerate(tl):
#         axs[1].scatter(t, param["muQ_l"][idx_slices[i]][1], c='red')
#         axs[1].scatter(t, param["muQ_r"][idx_slices[i]][1], c='orange')
#     axs[1].plot(x_hat_l[:, 1], c='blue')
#     axs[1].plot(x_hat_r[:, 1], c='green')
#     axs[1].set_ylabel("$x_2$")
#     axs[1].set_xlabel("$t$")
#     axs[1].set_xticks([0, cfg.nbData])
#     axs[1].set_xticklabels(["0", "T"])
#
#     dis_lst = []
#     for i in range(len(x_hat_l)):
#         dis_lst.append(np.sqrt(np.sum(np.square(x_hat_l[i, :2] - x_hat_r[i, :2]))))
#
#     dis_lst = np.array(dis_lst)
#     timestep = np.arange(len(dis_lst))
#     axs[2].plot(timestep, dis_lst)
#     axs[2].set_ylabel("traj_dis")
#     axs[2].set_xlabel("$t$")
#     axs[2].set_xticks([0, cfg.nbData])
#     axs[2].set_xticklabels(["0", "T"])
#
#     dis_lst = []
#     via_point_l = np.array(via_point_l)
#     via_point_r = np.array(via_point_r)
#     for i in range(len(via_point_l)):
#         dis_lst.append(np.sqrt(np.sum(np.square(via_point_l[i, :2] - via_point_r[i, :2]))))
#
#     dis_lst = np.array(dis_lst)
#     timestep = np.arange(len(dis_lst))
#     axs[3].plot(timestep, dis_lst)
#
#     plt.show()


def plot_3d_uni(x_hat, muQ=None, idx_slices=None, ori=False, save=False, save_file_name=None, g_ax=None, title=None,
                legend=None, for_test=False):
    if g_ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d', fc='white')
    else:
        ax = g_ax

    if muQ is not None and idx_slices is not None:
        for slice_t in idx_slices:
            ax.scatter(muQ[slice_t][0], muQ[slice_t][1], muQ[slice_t][2], c='red', s=10)

    if not isinstance(x_hat, list):
        if len(x_hat.shape) == 2:
            x_hat = np.expand_dims(x_hat, axis=0)

    title = 'Unimanual trajectory' if title is None else title
    rf.visualab.traj_plot(x_hat, legend=legend, title=title, mode='3d', ori=ori, g_ax=ax)

    if save:
        assert save_file_name is not None
        np.save(save_file_name, np.array(x_hat))
    if g_ax is None and not for_test:
        plt.show()


def plot_3d_bi(x_hat_l, x_hat_r, muQ_l=None, muQ_r=None, idx_slices=None, ori=False, save=False, save_file_name=None,
               g_ax=None, title=None, legend_lst=None, for_test=False):
    if g_ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d', fc='white')
    else:
        ax = g_ax

    if muQ_l is not None and muQ_r is not None and idx_slices is not None:
        for slice_t in idx_slices:
            ax.scatter(muQ_l[slice_t][0], muQ_l[slice_t][1], muQ_l[slice_t][2], c='red', s=10)
            ax.scatter(muQ_r[slice_t][0], muQ_r[slice_t][1], muQ_r[slice_t][2], c='orange', s=10)

    if not isinstance(x_hat_l, list):
        if len(x_hat_l.shape) == 2:
            x_hat_l = np.expand_dims(x_hat_l, axis=0)
            x_hat_r = np.expand_dims(x_hat_r, axis=0)

    title = 'Bimanual trajectory' if title is None else title
    legend_l = 'left arm' if legend_lst is None else legend_lst[0]
    legend_r = 'right arm' if legend_lst is None else legend_lst[1]
    rf.visualab.traj_plot(x_hat_l, title=title, legend=legend_l, mode='3d', ori=ori, g_ax=ax)
    rf.visualab.traj_plot(x_hat_r, legend=legend_r, mode='3d', ori=ori, g_ax=ax)

    if save:
        assert save_file_name is not None
        np.save(save_file_name[0], np.array(x_hat_l))
        np.save(save_file_name[1], np.array(x_hat_r))
    if g_ax is None and not for_test:
        plt.show()
