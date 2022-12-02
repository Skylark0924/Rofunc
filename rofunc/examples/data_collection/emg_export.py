"""
Delsys EMG Export
=================

This example shows how to process and visualize the EMG data.
"""

import rofunc as rf
import numpy as np
import matplotlib.pyplot as plt

emg = np.load('/home/ubuntu/Data/emg_record/20221202_181154.npy')
SAMPING_RATE = 2000
k = 4
n = 4
data_filter, data_clean, data_mvc, data_abs = rf.emg.process_all_channels(emg, n, SAMPING_RATE, k)

for i in range(n):
    rf.emg.plot_raw_and_clean(data_filter[:, i], data_clean[:, i], k)
    rf.emg.plot_abs_and_mvc(data_abs[:, i], data_mvc[:, i], k)
plt.show()

# # process single channel
# data_filter_1, data_clean_1, data_mvc_1, data_abs_1 = process(emg[:, 0], SAMPING_RATE, n)
# data_filter_2, data_clean_2, data_mvc_2, data_abs_2 = process(emg[:, 1], SAMPING_RATE, n)