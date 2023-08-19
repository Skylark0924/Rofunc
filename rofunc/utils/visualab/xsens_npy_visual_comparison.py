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
    # pos_x = filter(data[:, 0], 10)
    # pos_y = filter(data[:, 1], 10)
    # pos_z = filter(data[:, 2], 10)
    pos_x = data[:, 0]
    pos_y = data[:, 1]
    pos_z = data[:, 2]
    vel_x = [0] * len(data)
    vel_y = [0] * len(data)
    vel_z = [0] * len(data)
    for i in range(1, len(data)):
        vel_x[i - 1] = (data[i, 0] - data[i - 1, 0]) * 60
        vel_y[i - 1] = (data[i, 1] - data[i - 1, 1]) * 60
        vel_z[i - 1] = (data[i, 2] - data[i - 1, 2]) * 60
    # vel_x = filter(np.array(vel_x), 10)
    # vel_y = filter(np.array(vel_y), 10)
    # vel_z = filter(np.array(vel_z), 10)
    vel_x = np.array(vel_x)
    vel_y = np.array(vel_y)
    vel_z = np.array(vel_z)
    acc_x = [0] * len(data)
    acc_y = [0] * len(data)
    acc_z = [0] * len(data)
    for i in range(1, len(data)):
        acc_x[i - 1] = (vel_x[i] - vel_x[i - 1]) * 60
        acc_y[i - 1] = (vel_y[i] - vel_y[i - 1]) * 60
        acc_z[i - 1] = (vel_z[i] - vel_z[i - 1]) * 60
    # acc_x = filter(np.array(acc_x), 10)
    # acc_y = filter(np.array(acc_y), 10)
    # acc_z = filter(np.array(acc_z), 10)
    acc_x = np.array(acc_x)
    acc_y = np.array(acc_y)
    acc_z = np.array(acc_z)
    data_processed = np.array([pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z])
    return data_processed


data_1 = np.load('/home/ubuntu/Xsens_data/HKSI/20230412/xsens_mvnx/SQ_80kg/LeftHand.npy')
data_2 = np.load('/home/ubuntu/Xsens_data/HKSI/20230412/xsens_mvnx/SQ_90kg/LeftHand.npy')
data_3 = np.load('/home/ubuntu/Xsens_data/HKSI/20230412/xsens_mvnx/SQ_100kg/LeftHand.npy')
data_4 = np.load('/home/ubuntu/Xsens_data/HKSI/20230412/xsens_mvnx/SQ_110kg/LeftHand.npy')
data_5 = np.load('/home/ubuntu/Xsens_data/HKSI/20230412/xsens_mvnx/SQ_120kg_1/LeftHand.npy')
data_6 = np.load('/home/ubuntu/Xsens_data/HKSI/20230412/xsens_mvnx/SQ_120kg_2/LeftHand.npy')

# Bench_Press_0412
# data_processed_1 = process(data_1)[:, 1300:1480]
# data_processed_2 = process(data_2)[:, 1540:1720]
# data_processed_3 = process(data_3)[:, 650:830]
# data_processed_4 = process(data_4)[:, 1735:1955]
# data_processed_5 = process(data_5)[:, 720:1080]

# Dead_Lift_0412
# data_processed_1 = process(data_1)[:, 1920:2220]
# data_processed_2 = process(data_2)[:, 650:950]
# data_processed_3 = process(data_3)[:, 300:600]
# data_processed_4 = process(data_4)[:, 670:1000]

# Squat_0412
data_processed_1 = process(data_1)[:, 1160:1400]
data_processed_2 = process(data_2)[:, 730:970]
data_processed_3 = process(data_3)[:, 670:910]
data_processed_4 = process(data_4)[:, 590:830]
data_processed_5 = process(data_5)[:, 750:990]
data_processed_6 = process(data_6)[:, 1150:1390]

# t = np.arange(0, len(pos_y)/60, 1/60)
t = np.arange(0, 240 / 60, 1 / 60)
t_1 = np.arange(0, 330 / 60, 1 / 60)
t_2 = np.arange(0, 220 / 60, 1 / 60)

plt.figure(figsize=(20, 10))
plt.subplot(3, 3, 1)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.ylabel('pos (m)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed_1[0, :], color="royalblue", label='80kg', linewidth=1.5)
plt.plot(t, data_processed_2[0, :], color="orange", label='9kg', linewidth=1.5)
plt.plot(t, data_processed_3[0, :], color="red", label='100kg', linewidth=1.5)
plt.plot(t, data_processed_4[0, :], color="green", label='110kg', linewidth=1.5)
plt.plot(t, data_processed_5[0, :], color="black", label='120kg_failed', linewidth=1.5)
plt.plot(t, data_processed_6[0, :], color="purple", label='120kg', linewidth=1.5)
plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 2)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed_1[1, :], color="royalblue", label='80kg', linewidth=1.5)
plt.plot(t, data_processed_2[1, :], color="orange", label='9kg', linewidth=1.5)
plt.plot(t, data_processed_3[1, :], color="red", label='100kg', linewidth=1.5)
plt.plot(t, data_processed_4[1, :], color="green", label='110kg', linewidth=1.5)
plt.plot(t, data_processed_5[1, :], color="black", label='120kg_failed', linewidth=1.5)
plt.plot(t, data_processed_6[1, :], color="purple", label='120kg', linewidth=1.5)
# plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 3)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed_1[2, :], color="royalblue", label='80kg', linewidth=1.5)
plt.plot(t, data_processed_2[2, :], color="orange", label='9kg', linewidth=1.5)
plt.plot(t, data_processed_3[2, :], color="red", label='100kg', linewidth=1.5)
plt.plot(t, data_processed_4[2, :], color="green", label='110kg', linewidth=1.5)
plt.plot(t, data_processed_5[2, :], color="black", label='120kg_failed', linewidth=1.5)
plt.plot(t, data_processed_6[2, :], color="purple", label='120kg', linewidth=1.5)
# plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 4)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.ylabel('vel (m/s)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed_1[3, :], color="royalblue", label='80kg', linewidth=1.5)
plt.plot(t, data_processed_2[3, :], color="orange", label='9kg', linewidth=1.5)
plt.plot(t, data_processed_3[3, :], color="red", label='100kg', linewidth=1.5)
plt.plot(t, data_processed_4[3, :], color="green", label='110kg', linewidth=1.5)
plt.plot(t, data_processed_5[3, :], color="black", label='120kg_failed', linewidth=1.5)
plt.plot(t, data_processed_6[3, :], color="purple", label='120kg', linewidth=1.5)
# plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 5)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m/s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed_1[4, :], color="royalblue", label='80kg', linewidth=1.5)
plt.plot(t, data_processed_2[4, :], color="orange", label='9kg', linewidth=1.5)
plt.plot(t, data_processed_3[4, :], color="red", label='100kg', linewidth=1.5)
plt.plot(t, data_processed_4[4, :], color="green", label='110kg', linewidth=1.5)
plt.plot(t, data_processed_5[4, :], color="black", label='120kg_failed', linewidth=1.5)
plt.plot(t, data_processed_6[4, :], color="purple", label='120kg', linewidth=1.5)
# plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 6)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m/s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed_1[5, :], color="royalblue", label='80kg', linewidth=1.5)
plt.plot(t, data_processed_2[5, :], color="orange", label='9kg', linewidth=1.5)
plt.plot(t, data_processed_3[5, :], color="red", label='100kg', linewidth=1.5)
plt.plot(t, data_processed_4[5, :], color="green", label='110kg', linewidth=1.5)
plt.plot(t, data_processed_5[5, :], color="black", label='120kg_failed', linewidth=1.5)
plt.plot(t, data_processed_6[5, :], color="purple", label='120kg', linewidth=1.5)
# plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 7)
plt.xlabel('time (s)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.ylabel('acc (m/s^2)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed_1[6, :], color="royalblue", label='80kg', linewidth=1.5)
plt.plot(t, data_processed_2[6, :], color="orange", label='9kg', linewidth=1.5)
plt.plot(t, data_processed_3[6, :], color="red", label='100kg', linewidth=1.5)
plt.plot(t, data_processed_4[6, :], color="green", label='110kg', linewidth=1.5)
plt.plot(t, data_processed_5[6, :], color="black", label='120kg_failed', linewidth=1.5)
plt.plot(t, data_processed_6[6, :], color="purple", label='120kg', linewidth=1.5)
# plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 8)
plt.xlabel('time (s)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m/(s^2)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed_1[7, :], color="royalblue", label='80kg', linewidth=1.5)
plt.plot(t, data_processed_2[7, :], color="orange", label='9kg', linewidth=1.5)
plt.plot(t, data_processed_3[7, :], color="red", label='100kg', linewidth=1.5)
plt.plot(t, data_processed_4[7, :], color="green", label='110kg', linewidth=1.5)
plt.plot(t, data_processed_5[7, :], color="black", label='120kg_failed', linewidth=1.5)
plt.plot(t, data_processed_6[7, :], color="purple", label='120kg', linewidth=1.5)
# plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(3, 3, 9)
plt.xlabel('time (s)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m/(s^2)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(t, data_processed_1[8, :], color="royalblue", label='80kg', linewidth=1.5)
plt.plot(t, data_processed_2[8, :], color="orange", label='9kg', linewidth=1.5)
plt.plot(t, data_processed_3[8, :], color="red", label='100kg', linewidth=1.5)
plt.plot(t, data_processed_4[8, :], color="green", label='110kg', linewidth=1.5)
plt.plot(t, data_processed_5[8, :], color="black", label='120kg_failed', linewidth=1.5)
plt.plot(t, data_processed_6[8, :], color="purple", label='120kg', linewidth=1.5)
# plt.legend(loc="upper right", prop={'size': 12})

plt.suptitle("Squat", fontsize=30)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

plt.savefig('/home/ubuntu/Xsens_data/HKSI/20230412/fig/SQ_raw.png', dpi=300)
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
