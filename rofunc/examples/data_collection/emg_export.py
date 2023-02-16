"""
Delsys EMG Export
=================

This example shows how to process and visualize the EMG data.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import rofunc as rf


def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


emg = np.load('/home/ubuntu/Data/emg_record/20230216_052019.npy')
emg = emg[:, 1:]
SAMPING_RATE = 2000
k = 4
n = 6
data_filter, data_clean, data_mvc, data_abs = rf.emg.process_all_channels(emg, n, SAMPING_RATE, k)

for i in range(n):
    rf.emg.plot_raw_and_clean(data_filter[:, i], data_clean[:, i], k)
    rf.emg.plot_abs_and_mvc(data_abs[:, i], data_mvc[:, i], k)

filename = os.path.join('/home/ubuntu/Data/emg_record', '20230216_052019.pdf')
save_multi_image(filename)

plt.show()
