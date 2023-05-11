"""
Delsys EMG Export
=================

This example shows how to process and visualize the EMG data.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import math

import rofunc as rf

a = 20
b = 0.05


def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def impedance_profile(data):
    impedance_1 = np.zeros(len(data))
    impedance_2 = np.zeros(len(data))
    for i in range(len(data)):
        impedance_1[i] = a * (1 - math.exp(- b * (data[i, 0] + data[i, 1] + data[i, 2]))) / (
                    1 + math.exp(- b * (data[i, 0] + data[i, 1] + data[i, 2])))
    for i in range(len(data)):
        impedance_2[i] = 1.5 * a * (1 - math.exp(- b * (data[i, 3] + data[i, 3]))) / (
                1 + math.exp(- b * (data[i, 3] + data[i, 3])))
    return impedance_1, impedance_2


emg = np.load('/home/ubuntu/Data/emg_record/20230306/20230306_093814.npy')
emg = emg[:, 1:]
# emg = emg[:, :]
SAMPING_RATE = 2000
k = 4
n = 6
data_filter, data_clean, data_mvc, data_activity, data_abs, data_mvcscale = rf.emg.process_all_channels(emg, n,
                                                                                                        SAMPING_RATE, k)
impedance_1, impedance_2 = impedance_profile(data_mvcscale)
# impedance_1 = np.where(impedance_1 > 0.15, 1, 0)
# impedance_2 = np.where(impedance_2 > 0.15, 1, 0)
# plt.plot(impedance_1)
# plt.plot(impedance_2)
# plt.show()

for i in range(n):
    rf.emg.plot_raw_and_clean(data_filter[:, i], data_clean[:, i], k)
    rf.emg.plot_abs_and_mvc(data_abs[:, i], data_mvc[:, i], k)
    # rf.emg.plot_mvcscale_and_activity(data_mvcscale[:, i], data_activity[:, i], k)

# filename = os.path.join('/home/ubuntu/Data/emg_record', '20221221_094445.pdf')
# save_multi_image(filename)

plt.show()
