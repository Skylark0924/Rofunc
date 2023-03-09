import os

import numpy as np
import rofunc as rf


def test_7d_uni_lqt():
    via_points = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LQT_LQR/rolling_pin_1.npy'))
    filter_indices = [0, 1, 5, 10, 22, 36]
    via_points = via_points[filter_indices]

    controller = rf.planning.lqt.LQT(via_points)
    u_hat, x_hat, mu, idx_slices = controller.solve()
    rf.lqt.plot_3d_uni(x_hat, mu, idx_slices, ori=False, save=False)


# <editor-fold desc="7-dim Bi example">
# via_points = np.loadtxt(files('rofunc.data.LQT_LQR').joinpath('coffee_stirring_1.txt'), delimiter=', ')
# l_via_points = via_points[0:len(via_points):2]
# r_via_points = via_points[1:len(via_points):2]
# u_hat_l, u_hat_r, x_hat_l, x_hat_r, muQ_l, muQ_r, idx_slices = rf.lqt.bi(l_via_points, r_via_points)
# rf.lqt.plot_3d_bi(x_hat_l, x_hat_r, muQ_l, muQ_r, idx_slices, ori=False, save=False)
# </editor-fold>

# <editor-fold desc="7-dim Hierarchical example">
# via_points_raw = np.load(files('rofunc.data').joinpath('taichi_1l.npy'))
# filter_indices = [i for i in range(0, len(via_points_raw) - 10, 5)]
# filter_indices.append(len(via_points_raw) - 1)
# via_points_raw = via_points_raw[filter_indices]
# u_hat, x_hat, mu, idx_slices = rf.lqt.uni_hierarchical(via_points_raw, interval=2)
# rf.lqt.plot_3d_uni([x_hat], ori=False, save=False)
# </editor-fold>

if __name__ == '__main__':
    test_7d_uni_lqt()
