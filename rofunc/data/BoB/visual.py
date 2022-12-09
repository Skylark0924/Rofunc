import pandas as pd
import os
import rofunc as rf
import csv
import numpy as np

demo_path = os.path.join(input_path, demo_csv)
out_file_path = os.path.join(out_path, demo_csv)
rf.utils.delete_lines(demo_path, out_file_path, 14)
data = pd.read_csv("/home/ubuntu/BoB_data/2022-12-03/BoB data/bench_press/all_muscle_forces.csv", index_col=0)
# data = np.array(data)




print(data)