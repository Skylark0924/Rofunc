"""
BoB Visualize
================

This example shows how to use the muscle force files to visualize the muscle force.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rofunc as rf

input_path = '/home/lee/BoB_data/2022-12-03/BoB data/bench_press/'
out_path = os.path.join(input_path, 'process')
rf.oslab.create_dir(out_path)
demo_path = os.path.join(input_path, "all_muscle_forces.csv")
out_file_path = os.path.join(out_path, "all_muscle_forces_processed.csv")
rf.oslab.delete_lines(demo_path, out_file_path, 1)
data = pd.read_csv(out_file_path, index_col=0)

# Pectoralis major clavicular; Pectoralis major sternocostal; Deltoideus clavicular; Deltoideus scapular;
# Biceps femoris; Biceps femoris caput breve;
muscle_name = 'Biceps femoris caput breve right'
muscle_force_data = data.loc[:, muscle_name]
n = 8
muscle_force_data_process = np.convolve(muscle_force_data, np.ones((n,)) / n, mode='same')
t = np.arange(0, len(muscle_force_data_process) / 10, 0.1, float)
plt.plot(t, muscle_force_data_process)
plt.title(muscle_name)
plt.xlabel("t")
plt.ylabel("muscle force")
plt.savefig(os.path.join(out_path, muscle_name))
plt.show()
