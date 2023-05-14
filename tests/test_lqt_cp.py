import os

import numpy as np
import rofunc as rf


def test_7d_uni_cp_lqt():
    via_points = np.load('../data/taichi_1l.npy')
    filter_indices = [i for i in range(0, len(via_points) - 10, 5)]
    filter_indices.append(len(via_points) - 1)
    via_points = via_points[filter_indices]

    controller = rf.planning_control.lqt.LQTCP(via_points)
    u_hat, x_hat, mu, idx_slices = controller.solve()

    rf.lqt.plot_3d_uni(x_hat, mu, idx_slices, for_test=True)


if __name__ == '__main__':
    test_7d_uni_cp_lqt()
