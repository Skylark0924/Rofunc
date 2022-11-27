import rofunc as rf
import numpy as np
import os


def test_7d_uni_cp_lqt():
    via_points_raw = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/taichi_1l.npy'))
    filter_indices = [i for i in range(0, len(via_points_raw) - 10, 5)]
    filter_indices.append(len(via_points_raw) - 1)
    via_points_raw = via_points_raw[filter_indices]
    u_hat, x_hat, mu, idx_slices = rf.lqt.uni_cp(via_points_raw)
    rf.lqt.plot_3d_uni([x_hat], mu, idx_slices, for_test=True)


if __name__ == '__main__':
    test_7d_uni_cp_lqt()
