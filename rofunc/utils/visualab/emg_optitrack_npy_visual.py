"""
Warning: This file is deprecated and will be removed in a future release.
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import rofunc as rf

a = 20
b = 0.05

def impedance_profile(data):
    impedance_1 = np.zeros(len(data))
    impedance_2 = np.zeros(len(data))
    for i in range(len(data)):
        impedance_1[i] = 1.5 * a * (1 - math.exp(- b * (0.5 * data[i, 1] + 1.5 * data[i, 2]))) / (1 + math.exp(- b * (0.5 * data[i, 1] + 1.5 * data[i, 2])))
    for i in range(len(data)):
        impedance_2[i] = 1.5 * a * (1 - math.exp(- b * (1.5*data[i, 0] + 0.2*data[i, 2]))) / (
                    1 + math.exp(- b * (1.5*data[i, 0] + 0.2*data[i, 2])))
    # for i in range(len(data)):
    #     impedance_1[i] = 1.5 * a * (1 - math.exp(- b * (0.5 * data[i, 4] + 1.5 * data[i, 5]))) / (1 + math.exp(- b * (0.5 * data[i, 4] + 1.5 * data[i, 5])))
    # for i in range(len(data)):
    #     impedance_2[i] = 1.5 * a * (1 - math.exp(- b * (1.5*data[i, 3] + 0.2*data[i, 5]))) / (
    #                 1 + math.exp(- b * (1.5*data[i, 3] + 0.2*data[i, 5])))
    return impedance_1, impedance_2

emg = np.load('/home/lee/EMG_data/20230630/emg/20230630_161558.npy')
# stiffness = np.loadtxt("/home/ubuntu/Downloads/stiffness_50.txt")
# emg = emg[12000:50000, 1:4] / 100
emg = emg[:22000, 1:] / 50
np.place(emg, emg > 1, [0])
np.place(emg, emg < -1, [0])
# emg = emg[:, :]
SAMPING_RATE = 2000
k = 25
n = 6
data_filter, data_clean, data_mvc, data_activity, data_abs, data_mvcscale = rf.emg.process_all_channels(emg, n, SAMPING_RATE, k)
# data_clean = data_clean[650:1300, :]
# data_mvcscale = data_mvcscale[650:1300, :]
data_clean = data_clean[:, :]
data_mvcscale = data_mvcscale[:, :]
impedance_1, impedance_2 = impedance_profile(data_mvcscale)
t_1 = np.arange(0, len(impedance_2) * k / SAMPING_RATE, k / SAMPING_RATE)
t_2 = np.arange(0, 1, 0.02)
# impedance_1 = np.where(impedance_1 > 0.15, 1, 0)
impedance_threshold_1 = []
for i in range(len(impedance_1)):
    if impedance_1[i] > 0.5:
        impedance_threshold = 1
    elif impedance_1[i] > 0.2:
        impedance_threshold = 0.5
    else:
        impedance_threshold = 0
    impedance_threshold_1.append(impedance_threshold)
impedance_threshold_1 = np.array(impedance_threshold_1)

impedance_threshold_2 = []
for i in range(len(impedance_2)):
    if impedance_2[i] > 0.5:
        impedance_threshold = 1
    elif impedance_2[i] > 0.2:
        impedance_threshold = 0.5
    else:
        impedance_threshold = 0
    impedance_threshold_2.append(impedance_threshold)
impedance_threshold_2 = np.array(impedance_threshold_2)

# dance_threshold_1 = np.where(impedance_1 > 0.3, 1, 0)
# impedance_threshold_2 = np.where(impedance_2 > 0.3, 1, 0)

# data = np.load('/home/ubuntu/Data/optitrack/rigid_body.npy')
# left_hand_position = data[240:1440, 1, 6] / 100
# t_2 = np.arange(0, len(left_hand_position) / 120, 1 / 120)
# plt.plot(left_hand_position[:, 2])
# plt.show()

fig = plt.figure(facecolor='white', edgecolor='black', figsize=(20, 12))

# ---------------------20230306
# plt.subplot(2, 1, 1)
# # plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.xticks(fontproperties='Times New Roman', size=12)
# plt.yticks(fontproperties='Times New Roman', size=12)
# plt.plot(t_2, left_hand_position, color="royalblue", label='pos_z', linewidth=1.5)
# plt.legend(loc="upper right", prop={'size': 12})

# plt.subplot(4, 2, 1)
# # plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# # plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.xticks(fontproperties='Times New Roman', size=14)
# plt.yticks(fontproperties='Times New Roman', size=14)
# plt.ylim(-1.5, 2.5)
# plt.plot(t_1, data_clean[:, 3], color="gold", label='Processed EMG of AD_R', linewidth=2)
# plt.plot(t_1, data_mvcscale[:, 3], color="royalblue", label='Muscle Activation Level', linewidth=3)
# plt.legend(loc="upper right", prop={'size': 12})
# plt.text(3.3, 3.5, r'Signal Processing and Impedance Estimation', fontweight="bold", fontsize=22)
# plt.text(2, 2.7, r'Upper Body Right', fontsize=18)
# plt.text(10, 2.7, r'Upper Body Left', fontsize=18)
#
# plt.subplot(4, 2, 3)
# # plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# # plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.xticks(fontproperties='Times New Roman', size=14)
# plt.yticks(fontproperties='Times New Roman', size=14)
# plt.ylim(-1.5, 2.5)
# plt.plot(t_1, data_clean[:, 4], color="gold", label='Processed EMG of BB_R', linewidth=2)
# plt.plot(t_1, data_mvcscale[:, 4], color="royalblue", label='Muscle Activation Level', linewidth=3)
# plt.legend(loc="upper right", prop={'size': 12})
# # plt.text(5, 2, r'BB_R', fontsize=14)
#
# plt.subplot(4, 2, 5)
# # plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# # plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.xticks(fontproperties='Times New Roman', size=14)
# plt.yticks(fontproperties='Times New Roman', size=14)
# plt.ylim(-1.5, 2.5)
# plt.plot(t_1, data_clean[:, 5], color="gold", label='Processed EMG of TBLH_R', linewidth=2)
# plt.plot(t_1, data_mvcscale[:, 5], color="royalblue", label='Muscle Activation Level', linewidth=3)
# plt.legend(loc="upper right", prop={'size': 12})
# # plt.text(5, 2, r'TBLH_R', fontsize=14)
#
# plt.subplot(4, 2, 7)
# plt.xlabel('Time (s)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# # plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.xticks(fontproperties='Times New Roman', size=14)
# plt.yticks(fontproperties='Times New Roman', size=14)
# plt.ylim(-0.5, 2.5)
# plt.plot(t_1, impedance_2[:], color="limegreen", label='K_h of End Effector Right', linewidth=3)
# plt.plot(t_1, impedance_threshold_2[:], color="red", label='Threshold', linewidth=2)
# plt.legend(loc="upper right", prop={'size': 12})
# # plt.text(5, 2, r'End Effector_R', fontsize=14)

# plt.subplot(4, 2, 2)
# # plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# # plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.xticks(fontproperties='Times New Roman', size=14)
# plt.yticks(fontproperties='Times New Roman', size=14)
# plt.ylim(-1.5, 2.5)
# plt.plot(t_1, data_clean[:, 0], color="gold", label='Processed EMG of AD_L', linewidth=2)
# plt.plot(t_1, data_mvcscale[:, 0], color="royalblue", label='Muscle Activation Level', linewidth=3)
# plt.legend(loc="upper right", prop={'size': 12})
# # plt.text(5, 2, r'AD_L', fontsize=14)
#
# plt.subplot(4, 2, 4)
# # plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# # plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.xticks(fontproperties='Times New Roman', size=14)
# plt.yticks(fontproperties='Times New Roman', size=14)
# plt.ylim(-1.5, 2.5)
# plt.plot(t_1, data_clean[:, 1], color="gold", label='Processed EMG of BB_L', linewidth=2)
# plt.plot(t_1, data_mvcscale[:, 1], color="royalblue", label='Muscle Activation Level', linewidth=3)
# plt.legend(loc="upper right", prop={'size': 12})
# # plt.text(5, 2, r'BB_L', fontsize=14)
#
# plt.subplot(4, 2, 6)
# # plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# # plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.xticks(fontproperties='Times New Roman', size=14)
# plt.yticks(fontproperties='Times New Roman', size=14)
# plt.ylim(-1.5, 2.5)
# plt.plot(t_1, data_clean[:, 2], color="gold", label='Processed EMG of TBLH_L', linewidth=2)
# plt.plot(t_1, data_mvcscale[:, 2], color="royalblue", label='Muscle Activation Level', linewidth=3)
# plt.legend(loc="upper right", prop={'size': 12})
# # plt.text(5, 2, r'TBLH_L', fontsize=14)
#
# plt.subplot(4, 2, 8)
# plt.xlabel('Time (s)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# # plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.xticks(fontproperties='Times New Roman', size=14)
# plt.yticks(fontproperties='Times New Roman', size=14)
# plt.ylim(-0.5, 2.5)
# plt.plot(t_1, impedance_1[:], color="limegreen", label='K_h of End Effector Left', linewidth=3)
# plt.plot(t_1, impedance_threshold_1[:], color="red", label='Threshold', linewidth=2)
# # plt.text(5, 2, r'End Effector_Left', fontsize=14)
# plt.legend(loc="upper right", prop={'size': 12})

# ---------------------20230627
plt.subplot(2, 3, 1).patch.set(facecolor='white')
plt.subplot(2, 3, 1).spines['top'].set_color('black')
plt.subplot(2, 3, 1).spines['bottom'].set_color('black')
plt.subplot(2, 3, 1).spines['left'].set_color('black')
plt.subplot(2, 3, 1).spines['right'].set_color('black')
plt.xlabel('t', fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=16)
plt.yticks(fontproperties='Times New Roman', size=16)
plt.ylim(-2.5, 2.5)
plt.plot(t_1, data_clean[:, 0], color="gold", label='Processed EMG Signals', linewidth=1.5)
plt.plot(t_1, data_mvcscale[:, 0], color="royalblue", label='Muscle Activation Level', linewidth=3)
# plt.plot(t_1, data_clean[:, 3], color="gold", label='Processed EMG Signals', linewidth=1.5)
# plt.plot(t_1, data_mvcscale[:, 3], color="royalblue", label='Muscle Activation Level', linewidth=3)

plt.subplot(2, 3, 2).patch.set(facecolor='white')
plt.subplot(2, 3, 2).spines['top'].set_color('black')
plt.subplot(2, 3, 2).spines['bottom'].set_color('black')
plt.subplot(2, 3, 2).spines['left'].set_color('black')
plt.subplot(2, 3, 2).spines['right'].set_color('black')
plt.xlabel('t', fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=16)
plt.yticks(fontproperties='Times New Roman', size=16)
plt.ylim(-2.5, 2.5)
plt.plot(t_1, data_clean[:, 1], color="gold", label='Processed EMG Signals', linewidth=1.5)
plt.plot(t_1, data_mvcscale[:, 1], color="royalblue", label='Muscle Activation Level', linewidth=3)
# plt.plot(t_1, data_clean[:, 4], color="gold", label='Processed EMG Signals', linewidth=1.5)
# plt.plot(t_1, data_mvcscale[:, 4], color="royalblue", label='Muscle Activation Level', linewidth=3)
plt.legend(loc="upper left", facecolor='white', edgecolor='grey', prop={'size': 18})

plt.subplot(2, 3, 3).patch.set(facecolor='white')
plt.subplot(2, 3, 3).spines['top'].set_color('black')
plt.subplot(2, 3, 3).spines['bottom'].set_color('black')
plt.subplot(2, 3, 3).spines['left'].set_color('black')
plt.subplot(2, 3, 3).spines['right'].set_color('black')
plt.xlabel('t', fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=16)
plt.yticks(fontproperties='Times New Roman', size=16)
plt.ylim(-2.5, 2.5)
plt.plot(t_1, data_clean[:, 2], color="gold", linewidth=1.5)
plt.plot(t_1, data_mvcscale[:, 2], color="royalblue", linewidth=3)
# plt.plot(t_1, data_clean[:, 5], color="gold", linewidth=1.5)
# plt.plot(t_1, data_mvcscale[:, 5], color="royalblue", linewidth=3)

plt.subplot(2, 3, 4).patch.set(facecolor='white')
plt.subplot(2, 3, 4).spines['top'].set_color('black')
plt.subplot(2, 3, 4).spines['bottom'].set_color('black')
plt.subplot(2, 3, 4).spines['left'].set_color('black')
plt.subplot(2, 3, 4).spines['right'].set_color('black')
plt.xlabel('t', fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=16)
plt.yticks(fontproperties='Times New Roman', size=16)
plt.ylim(-0.5, 2.5)
plt.plot(t_1, impedance_2[:], color="black", label='Estimated Stiffness $K_{y}^{h}$', linewidth=3)
plt.plot(t_1, impedance_threshold_2[:], color="red", label='Threshold', linewidth=2)
plt.legend(loc="upper left", facecolor='white', edgecolor='grey', prop={'size': 18})

plt.subplot(2, 3, 5).patch.set(facecolor='white')
plt.subplot(2, 3, 5).spines['top'].set_color('black')
plt.subplot(2, 3, 5).spines['bottom'].set_color('black')
plt.subplot(2, 3, 5).spines['left'].set_color('black')
plt.subplot(2, 3, 5).spines['right'].set_color('black')
plt.xlabel('t', fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=16)
plt.yticks(fontproperties='Times New Roman', size=16)
plt.ylim(-0.5, 2.5)
plt.plot(t_1, impedance_1[:], color="black", label='Estimated Stiffness $K_{z}^{h}$', linewidth=3)
plt.plot(t_1, impedance_threshold_1[:], color="red", label='Threshold', linewidth=2)
plt.legend(loc="upper left", facecolor='white', edgecolor='grey', prop={'size': 18})

# plt.subplot(2, 3, 5).patch.set(facecolor='white')
# plt.subplot(2, 3, 5).spines['top'].set_color('black')
# plt.subplot(2, 3, 5).spines['bottom'].set_color('black')
# plt.subplot(2, 3, 5).spines['left'].set_color('black')
# plt.subplot(2, 3, 5).spines['right'].set_color('black')
# plt.xlabel('trajectory', fontdict={'family': 'Times New Roman'}, fontsize=18)
# # plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.xticks(fontproperties='Times New Roman', size=14)
# plt.yticks(fontproperties='Times New Roman', size=14)
# plt.ylim(0, 800)
# plt.plot(t_2, stiffness, color="black", label='Robot Stiffness $K$ ($N/m$)', linewidth=3)
# # plt.text(5, 2, r'End Effector_Left', fontsize=14)
# plt.legend(loc="upper left", facecolor='white', edgecolor='grey', prop={'size': 18})
#
# plt.subplot(2, 3, 6).patch.set(facecolor='white')
# plt.subplot(2, 3, 6).spines['top'].set_color('black')
# plt.subplot(2, 3, 6).spines['bottom'].set_color('black')
# plt.subplot(2, 3, 6).spines['left'].set_color('black')
# plt.subplot(2, 3, 6).spines['right'].set_color('black')
# plt.xlabel('trajectory', fontdict={'family': 'Times New Roman'}, fontsize=18)
# # plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.xticks(fontproperties='Times New Roman', size=14)
# plt.yticks(fontproperties='Times New Roman', size=14)
# plt.ylim(10, 70)
# plt.plot(t_2, 2 * np.sqrt(stiffness), color="black", label='Robot Damping $D$ ($N/m^2$)', linewidth=3)
# # plt.text(5, 2, r'End Effector_Left', fontsize=14)
# plt.legend(loc="upper left", facecolor='white', edgecolor='grey', prop={'size': 18})

# plt.suptitle("EMG Signal Processing and Impedance Estimation of Demonstrations", fontsize=20)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.2)

np.save('/home/lee/EMG_data/20230630/impedance data/5_left_impedance_z.npy', impedance_1)
np.save('/home/lee/EMG_data/20230630/impedance data/5_left_impedance_y.npy', impedance_2)
plt.savefig('/home/lee/EMG_data/20230630/fig/5_left.png', dpi=600)
plt.show()

