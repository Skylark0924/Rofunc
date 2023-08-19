"""
Warning: This file is deprecated and will be removed in a future release.
"""

import matplotlib.pyplot as plt
import numpy as np


# data_1 = np.load('/home/lee/EMG_data/20230630/impedance data/1_left_impedance_y.npy')
# data_2 = np.load('/home/lee/EMG_data/20230630/impedance data/2_left_impedance_y.npy')
# data_3 = np.load('/home/lee/EMG_data/20230630/impedance data/3_left_impedance_y.npy')
# data_4 = np.load('/home/lee/EMG_data/20230630/impedance data/4_left_impedance_y.npy')
# data_5 = np.load('/home/lee/EMG_data/20230630/impedance data/5_left_impedance_y.npy')
# data_6 = np.load('/home/lee/EMG_data/20230630/impedance data/6_left_impedance_y.npy')

data_1 = np.load('/home/lee/EMG_data/20230630/impedance data/1_left_impedance_z.npy')
data_2 = np.load('/home/lee/EMG_data/20230630/impedance data/2_left_impedance_z.npy')
data_3 = np.load('/home/lee/EMG_data/20230630/impedance data/3_left_impedance_z.npy')
data_4 = np.load('/home/lee/EMG_data/20230630/impedance data/4_left_impedance_z.npy')
data_5 = np.load('/home/lee/EMG_data/20230630/impedance data/5_left_impedance_z.npy')
data_6 = np.load('/home/lee/EMG_data/20230630/impedance data/6_left_impedance_z.npy')

plt.figure(figsize=(20, 10))
plt.subplot(1, 1, 1)
plt.xlabel('t', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
# plt.ylabel('pos (m)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
plt.xticks(fontproperties='Times New Roman', size=12)
plt.yticks(fontproperties='Times New Roman', size=12)
plt.plot(np.arange(0, 1, 1/len(data_1)), data_1, color="royalblue", label='1', linewidth=1.5)
# plt.plot(np.arange(0, 1, 1/len(data_2)), data_2, color="orange", label='2', linewidth=1.5)
# plt.plot(np.arange(0, 1, 1/len(data_3)), data_3, color="red", label='3', linewidth=1.5)
plt.plot(np.arange(0, 1, 1/len(data_4)), data_4, color="green", label='4', linewidth=1.5)
# plt.plot(np.arange(0, 1, 1/len(data_5)), data_5, color="black", label='5', linewidth=1.5)
plt.plot(np.arange(0, 1, 1/len(data_6)), data_6, color="purple", label='6', linewidth=1.5)
plt.legend(loc="upper right", prop={'size': 12})


# plt.suptitle("Squat", fontsize=30)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

# plt.savefig('/home/ubuntu/Xsens_data/HKSI/20230412/fig/SQ_raw.png', dpi=300)
plt.show()

