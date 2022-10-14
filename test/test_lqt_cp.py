import rofunc as rf
import numpy as np
from importlib_resources import files

via_points_raw = np.load(files('rofunc.data').joinpath('taichi_1l.npy'))
filter_indices = [i for i in range(0, len(via_points_raw) - 10, 5)]
filter_indices.append(len(via_points_raw) - 1)
via_points_raw = via_points_raw[filter_indices]
rf.lqt.uni_cp(via_points_raw)
