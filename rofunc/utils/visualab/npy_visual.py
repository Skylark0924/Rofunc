import matplotlib.pyplot as plt
import numpy as np

data = np.load('/home/ubuntu/Xsens_data/HKSI/bench_press#Chenzui/RightHand.npy')
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
t = np.arange(0, 3200/60, 1/60)

# fig, ax0 = plt.subplots(nrows=1, ncols=1, sharex=True)
# ax0.set_xlabel("Time", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
# ax0.set_ylabel("(m)", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
# plt.subplots_adjust(hspace=0.2)
# legend_font = {"family": "Times New Roman"}
# ax0.plot(t, pos_z[400:3600], color="#B0BEC5", label="pos", zorder=1)
# ax1 = ax0.twinx()
# ax1.set_ylabel("(m/s)", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
# ax1.plot(t, vel_z[400:3600], color="#FA6839", label="vel", linewidth=1.5)
# ax0.legend(loc="upper right", frameon=True, prop=legend_font)
# ax1.legend(loc="upper left", frameon=True, prop=legend_font)
# plt.show()

fig, ax0 = plt.subplots(nrows=1, ncols=1, sharex=True)
ax0.set_ylim((-10, 10))
ax0.set_xlabel("Time", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
ax0.set_ylabel("(m/s^2)", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
plt.subplots_adjust(hspace=0.2)
legend_font = {"family": "Times New Roman"}
ax0.plot(t, acc_z[400:3600], color="#B0BEC5", label="acc", zorder=1)
ax0.legend(loc="upper right", frameon=True, prop=legend_font)
plt.show()