import pandas as pd
import csv
import numpy as np

with open("/home/ubuntu/BoB_data/2022-12-03/BoB data/bench_press/all_muscle_forces.csv", 'r') as x:
    sample_data = list(csv.reader(x, delimiter=","))
data = np.array(sample_data[1:])
a = data[1]



print(list)