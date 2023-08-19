"""
Warning: This file is deprecated and will be removed in a future release.
"""

import matplotlib.pyplot as plt
import numpy as np


def filter(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

def process(data):
    pos_x = filter(data[:, 0], 10)
    pos_y = filter(data[:, 1], 10)
    pos_z = filter(data[:, 2], 10)
    vel_x = [0] * len(data)
    vel_y = [0] * len(data)
    vel_z = [0] * len(data)
    for i in range(1, len(data)):
        vel_x[i - 1] = (data[i, 0] - data[i - 1, 0]) * 60
        vel_y[i - 1] = (data[i, 1] - data[i - 1, 1]) * 60
        vel_z[i - 1] = (data[i, 2] - data[i - 1, 2]) * 60
    vel_x = filter(np.array(vel_x), 10)
    vel_y = filter(np.array(vel_y), 10)
    vel_z = filter(np.array(vel_z), 10)
    acc_x = [0] * len(data)
    acc_y = [0] * len(data)
    acc_z = [0] * len(data)
    for i in range(1, len(data)):
        acc_x[i - 1] = (vel_x[i] - vel_x[i - 1]) * 60
        acc_y[i - 1] = (vel_y[i] - vel_y[i - 1]) * 60
        acc_z[i - 1] = (vel_z[i] - vel_z[i - 1]) * 60
    acc_x = filter(np.array(acc_x), 10)
    acc_y = filter(np.array(acc_y), 10)
    acc_z = filter(np.array(acc_z), 10)
    data_processed = np.array([pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z])
    return data_processed

data = np.load('/home/ubuntu/Xsens_data/HKSI/20230322/ForceEMG-012/RightHand.npy')
data_processed = process(data)[:, 100:900]
# t = np.arange(0, len(pos_y)/60, 1/60)
t = np.arange(0, 800/60, 1/60)

plt.figure(figsize=(20, 10))
plt.subplot(3, 3, 1)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed[0, :], color="royalblue", label='pos_x', linewidth=1.5)
plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 2)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed[1, :], color="royalblue", label='pos_y', linewidth=1.5)
plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 3)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed[2, :], color="royalblue", label='pos_z', linewidth=1.5)
plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 4)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.ylabel('m/s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed[3, :], color="darkorange", label='vel_x', linewidth=1.5)
plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 5)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m/s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed[4, :], color="darkorange", label='vel_y', linewidth=1.5)
plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 6)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m/s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed[5, :], color="darkorange", label='vel_z', linewidth=1.5)
plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 7)
plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.ylabel('m/(s^2)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed[6, :], color="lightcoral", label='acc_x', linewidth=1.5)
plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 8)
plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m/(s^2)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed[7, :], color="lightcoral", label='acc_y', linewidth=1.5)
plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 9)
plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m/(s^2)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed[8, :], color="lightcoral", label='acc_z', linewidth=1.5)
plt.legend(loc="upper right", prop={'size': 12})

plt.suptitle("BicepsCurl_90%_15kg", fontsize=30)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

# plt.savefig('/home/ubuntu/Xsens_data/HKSI/20230322/fig/11.png', dpi=300)
plt.show()

# fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), tight_layout=True)
# ax0.set_xlabel("Time", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
# ax0.set_ylabel("(m)", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
# plt.subplots_adjust(hspace=0.2)
# legend_font = {"family": "Times New Roman"}
# ax0.plot(t, pos_x[1300:2400], color="#B0BEC5", label="pos", zorder=1)
# ax1 = ax0.twinx()
# ax1.set_ylabel("(m/s)", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
# ax1.plot(t, vel_x[1300:2400], color="#FA6839", label="vel", linewidth=1.5)
# ax0.legend(loc="upper left", frameon=True, prop=legend_font)
# ax1.legend(loc="upper right", frameon=True, prop=legend_font)
# plt.savefig('/home/lee/Xsens_data/HKSI/figures_20230208/jump/pos_and_vel_x.png', dpi=300, bbox_inch='tight')
# plt.show()
#
# fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), tight_layout=True)
# ax0.set_xlabel("Time", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
# ax0.set_ylabel("(m/s^2)", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
# plt.subplots_adjust(hspace=0.2)
# legend_font = {"family": "Times New Roman"}
# ax0.plot(t, acc_x[1300:2400], color="#349BEB", label="acc", zorder=1)
# ax0.legend(loc="up:t", frameon=True, prop=legend_font)
# plt.savefig('/home/lee/Xsens_data/HKSI/figures_20230208/jump/acc_x.png', dpi=300, bbox_inch='tight')
# plt.show()