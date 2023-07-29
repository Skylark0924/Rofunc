import numpy as np

import rofunc as rf


def test_7d_uni_lqt():
    via_points = np.load('../examples/data/LQT_LQR/rolling_pin_1.npy')
    filter_indices = [0, 1, 5, 10, 22, 36]
    via_points = via_points[filter_indices]

    controller = rf.planning_control.lqt.LQT(via_points)
    u_hat, x_hat, mu, idx_slices = controller.solve()
    rf.lqt.plot_3d_uni(x_hat, mu, idx_slices, ori=False, save=False, for_test=True)


def test_7d_uni_lqt_hierarchical():
    via_points = np.load('../examples/data/LQT_LQR/taichi_1l.npy')
    filter_indices = [i for i in range(0, len(via_points) - 10, 5)]
    via_points = via_points[filter_indices]

    controller = rf.planning_control.lqt.LQTHierarchical(via_points, interval=3)
    u_hat, x_hat, mu, idx_slices = controller.solve()
    rf.lqt.plot_3d_uni(x_hat, ori=False, save=False, for_test=True)


def test_7d_uni_lqt_bi():
    all_points = np.genfromtxt('../examples/data/LQT_LQR/coffee_stirring_1.txt', delimiter=', ')
    all_points_l = all_points[0:len(all_points):2]
    all_points_r = all_points[1:len(all_points):2]

    controller = rf.planning_control.lqt.LQTBi(all_points_l, all_points_r)
    u_hat_l, u_hat_r, x_hat_l, x_hat_r, mu_l, mu_r, idx_slices = controller.solve()
    rf.lqt.plot_3d_bi(x_hat_l, x_hat_r, mu_l, mu_r, idx_slices, ori=False, save=False, for_test=True)


if __name__ == '__main__':
    test_7d_uni_lqt()
    test_7d_uni_lqt_hierarchical()
    test_7d_uni_lqt_bi()
