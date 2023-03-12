import numpy as np
import rofunc as rf


# def test_2d_fb_lqt():
#     # TODO: need to modify the definition of state noise
#     cfg = rf.config.utils.get_config("./planning", "lqt_2d")
#     via_points = np.array([[2, 5, 0, 0], [3, 1, 0, 0]])
#     state_noise = ...
#
#     controller = rf.lqt.LQTFb(via_points)
#     controller.solve(state_noise, for_test=True)


def test_7d_uni_fb_lqt():
    via_points = np.zeros((3, 14))
    via_points[0, :7] = np.array([2, 5, 3, 0, 0, 0, 1])
    via_points[1, :7] = np.array([3, 1, 1, 0, 0, 0, 1])
    via_points[2, :7] = np.array([5, 4, 1, 0, 0, 0, 1])
    cfg = rf.config.utils.get_config('./planning', 'lqt')
    state_noise = np.hstack((-1, -.2, 1, 0, 0, 0, 0, np.zeros(cfg.nbVarX - cfg.nbVarPos)))

    controller = rf.lqt.LQTFb(via_points)
    controller.solve(state_noise, for_test=True)


if __name__ == '__main__':
    test_7d_uni_fb_lqt()
