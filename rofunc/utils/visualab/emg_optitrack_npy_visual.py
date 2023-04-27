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
        impedance_1[i] = 1.5 * a * (1 - math.exp(- b * (1.5*data[i, 0] + 0.5*data[i, 2]))) / (1 + math.exp(- b * (1.5*data[i, 0] + 0.5*data[i, 2])))
    for i in range(len(data)):
        impedance_2[i] = 1.5 * a * (1 - math.exp(- b * (1.5*data[i, 3] + 0.5*data[i, 5]))) / (
                    1 + math.exp(- b * (1.5*data[i, 3] + 0.5*data[i, 5])))
    return impedance_1, impedance_2

emg = np.load('/home/ubuntu/Data/emg_record/20230306/20230306_095626.npy')
emg = emg[12000:50000, 1:] / 100
# emg = emg[:, :]
SAMPING_RATE = 2000
k = 20
n = 6
data_filter, data_clean, data_mvc, data_activity, data_abs, data_mvcscale = rf.emg.process_all_channels(emg, n, SAMPING_RATE, k)
data_clean = data_clean[500:1150, :]
data_mvcscale = data_mvcscale[500:1150, :]
# data_cleanscale = (data_clean - data_clean.min()) / (data_clean.max() - data_clean.min())
impedance_1, impedance_2 = impedance_profile(data_mvcscale)
t_1 = np.arange(0, len(impedance_2) * k / SAMPING_RATE, k / SAMPING_RATE)
# impedance_1 = np.where(impedance_1 > 0.15, 1, 0)
impedance_threshold_1 = np.where(impedance_1 > 0.25, 1, 0)
impedance_threshold_2 = np.where(impedance_2 > 0.3, 1, 0)
# plt.plot(impedance_1)
# plt.plot(impedance_2)
# plt.show()

data = np.load('/home/ubuntu/Data/optitrack/rigid_body.npy')
left_hand_position = data[240:1440, 1, 6] / 100
t_2 = np.arange(0, len(left_hand_position) / 120, 1 / 120)
# plt.plot(left_hand_position[:, 2])
# plt.show()

plt.figure(figsize=(20, 10))

# plt.subplot(2, 1, 1)
# # plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.xticks(fontproperties='Times New Roman', size=12)
# plt.yticks(fontproperties='Times New Roman', size=12)
# plt.plot(t_2, left_hand_position, color="royalblue", label='pos_z', linewidth=1.5)
# plt.legend(loc="upper right", prop={'size': 12})

plt.subplot(4, 2, 1)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=14)
plt.yticks(fontproperties='Times New Roman', size=14)
plt.ylim(-1.5, 2.5)
plt.plot(t_1, data_clean[:, 3], color="gold", label='Processed EMG of AD_R', linewidth=2)
plt.plot(t_1, data_mvcscale[:, 3], color="royalblue", label='Muscle Activation Level', linewidth=3)
plt.legend(loc="upper right", prop={'size': 12})
plt.text(3.3, 3.5, r'Signal Processing and Impedance Estimation', fontweight="bold", fontsize=22)
plt.text(2, 2.7, r'Upper Body Right', fontsize=18)
plt.text(10, 2.7, r'Upper Body Left', fontsize=18)

plt.subplot(4, 2, 3)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=14)
plt.yticks(fontproperties='Times New Roman', size=14)
plt.ylim(-1.5, 2.5)
plt.plot(t_1, data_clean[:, 4], color="gold", label='Processed EMG of BB_R', linewidth=2)
plt.plot(t_1, data_mvcscale[:, 4], color="royalblue", label='Muscle Activation Level', linewidth=3)
plt.legend(loc="upper right", prop={'size': 12})
# plt.text(5, 2, r'BB_R', fontsize=14)

plt.subplot(4, 2, 5)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=14)
plt.yticks(fontproperties='Times New Roman', size=14)
plt.ylim(-1.5, 2.5)
plt.plot(t_1, data_clean[:, 5], color="gold", label='Processed EMG of TBLH_R', linewidth=2)
plt.plot(t_1, data_mvcscale[:, 5], color="royalblue", label='Muscle Activation Level', linewidth=3)
plt.legend(loc="upper right", prop={'size': 12})
# plt.text(5, 2, r'TBLH_R', fontsize=14)

plt.subplot(4, 2, 7)
plt.xlabel('Time (s)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=14)
plt.yticks(fontproperties='Times New Roman', size=14)
plt.ylim(-0.5, 2.5)
plt.plot(t_1, impedance_2[:], color="limegreen", label='K_h of End Effector Right', linewidth=3)
plt.plot(t_1, impedance_threshold_2[:], color="red", label='Threshold', linewidth=2)
plt.legend(loc="upper right", prop={'size': 12})
# plt.text(5, 2, r'End Effector_R', fontsize=14)

plt.subplot(4, 2, 2)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=14)
plt.yticks(fontproperties='Times New Roman', size=14)
plt.ylim(-1.5, 2.5)
plt.plot(t_1, data_clean[:, 0], color="gold", label='Processed EMG of AD_L', linewidth=2)
plt.plot(t_1, data_mvcscale[:, 0], color="royalblue", label='Muscle Activation Level', linewidth=3)
plt.legend(loc="upper right", prop={'size': 12})
# plt.text(5, 2, r'AD_L', fontsize=14)

plt.subplot(4, 2, 4)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=14)
plt.yticks(fontproperties='Times New Roman', size=14)
plt.ylim(-1.5, 2.5)
plt.plot(t_1, data_clean[:, 1], color="gold", label='Processed EMG of BB_L', linewidth=2)
plt.plot(t_1, data_mvcscale[:, 1], color="royalblue", label='Muscle Activation Level', linewidth=3)
plt.legend(loc="upper right", prop={'size': 12})
# plt.text(5, 2, r'BB_L', fontsize=14)

plt.subplot(4, 2, 6)
# plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=14)
plt.yticks(fontproperties='Times New Roman', size=14)
plt.ylim(-1.5, 2.5)
plt.plot(t_1, data_clean[:, 2], color="gold", label='Processed EMG of TBLH_L', linewidth=2)
plt.plot(t_1, data_mvcscale[:, 2], color="royalblue", label='Muscle Activation Level', linewidth=3)
plt.legend(loc="upper right", prop={'size': 12})
# plt.text(5, 2, r'TBLH_L', fontsize=14)

plt.subplot(4, 2, 8)
plt.xlabel('Time (s)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=14)
plt.yticks(fontproperties='Times New Roman', size=14)
plt.ylim(-0.5, 2.5)
plt.plot(t_1, impedance_1[:], color="limegreen", label='K_h of End Effector Left', linewidth=3)
plt.plot(t_1, impedance_threshold_1[:], color="red", label='Threshold', linewidth=2)
# plt.text(5, 2, r'End Effector_Left', fontsize=14)
plt.legend(loc="upper right", prop={'size': 12})

# plt.suptitle("EMG Signal Processing and Impedance Estimation of Demonstrations", fontsize=20)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.2)

plt.savefig('/home/ubuntu/Data/emg_record/20230306/3.png', dpi=1000)
plt.show()

# fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), tight_layout=True)
# ax0.set_xlabel("Time", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
# ax0.set_ylabel("Scale", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
# plt.subplots_adjust(hspace=0.2)
# legend_font = {"family": "Times New Roman"}
# ax0.plot(t_1, impedance_2[:], color="#B0BEC5", label="impedance", zorder=1)
# ax1 = ax0.twinx()
# ax1.set_ylabel("m", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
# ax1.plot(t_2, left_hand_position[:], color="#FA6839", label="pos", linewidth=1.5)
# ax0.legend(loc="upper left", frameon=True, prop=legend_font)
# ax1.legend(loc="upper right", frameon=True, prop=legend_font)
# plt.show()
