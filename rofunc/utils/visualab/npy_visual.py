import matplotlib.pyplot as plt
import numpy as np

data = np.load('/home/ubuntu/Xsens_data/HKSI/bench_press#Chenzui/RightHand.npy')

vel_x = [0]*len(data)
vel_y = [0]*len(data)
vel_z = [0]*len(data)
for i in range(1, len(data)):
    vel_x[i-1] = (data[i, 0] - data[i-1, 0]) * 60
    vel_y[i-1] = (data[i, 1] - data[i - 1, 1]) * 60
    vel_z[i-1] = (data[i, 2] - data[i - 1, 2]) * 60
# plt.plot(data[0:4000, 0], 'r')
# plt.plot(vel_x[0:4000], 'b')
# plt.plot(data[0:4000, 1], 'b')
# plt.plot(data[0:4000, 2], label='pos',  color='green')
# plt.plot(vel_z[0:4000], label='vel', color='blue')
# plt.show()

fig, ax0 = plt.subplots(nrows=1, ncols=1, sharex=True)
ax0.set_xlabel("Time", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
ax0.set_ylabel("(m)", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
# fig.suptitle("Position and Velocity", fontweight="bold",
#              fontdict={'family': 'Times New Roman'}, fontsize=16)
plt.subplots_adjust(hspace=0.2)
# x_axis = np.linspace(0, data_abs.shape[0] / int(2000 / k), data_abs.shape[0])
legend_font = {"family": "Times New Roman"}
# ax0.set_title("Sensor", fontdict={'family': 'Times New Roman'}, fontsize=12)
ax0.plot(data[400:3600, 2], color="#B0BEC5", label="pos", zorder=1)
ax1 = ax0.twinx()
ax1.set_ylabel("(m/s)", fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=12)
ax1.plot(
    vel_z[400:3600], color="#FA6839", label="vel", linewidth=1.5
)
ax0.legend(loc="upper right", frameon=True, prop=legend_font)
ax1.legend(loc="upper left", frameon=True, prop=legend_font)
plt.show()