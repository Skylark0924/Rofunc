import rofunc as rf
import numpy as np
from importlib_resources import files


# <editor-fold desc="2-dim Uni example">
# demo_points = np.array([[[0, 0], [-1, 8], [4, 3], [2, 1], [4, 3]],
#                         [[0, -2], [-1, 7], [3, 2.5], [2, 1.6], [4, 3]],
#                         [[0, -1], [-1, 8], [4, 5.2], [2, 1.1], [4, 3.5]]])
# demos_x = rf.data_generator.multi_bezier_demos(demo_points)  # (3, 50, 2): 3 demos, each has 50 points
# model, rep = rf.tpgmm.uni(demos_x, show_demo_idx=2, plot=True)
# </editor-fold>

# <editor-fold desc="2-dim Bi example">
# left_demo_points = np.array([[[0, 0], [-1, 8], [4, 3], [2, 1], [4, 3]],
#                              [[0, -2], [-1, 7], [3, 2.5], [2, 1.6], [4, 3]],
#                              [[0, -1], [-1, 8], [4, 5.2], [2, 1.1], [4, 3.5]]])
# right_demo_points = np.array([[[8, 8], [7, 1], [4, 3], [6, 8], [4, 3]],
#                               [[8, 7], [7, 1], [3, 3], [6, 6], [4, 3]],
#                               [[8, 8], [7, 1], [4, 5], [6, 8], [4, 3.5]]])
# demos_left_x = rf.data_generator.multi_bezier_demos(left_demo_points)  # (3, 50, 2): 3 demos, each has 50 points
# demos_right_x = rf.data_generator.multi_bezier_demos(right_demo_points)
# model_l, model_r, rep_l, rep_r = rf.tpgmm.bi(demos_left_x, demos_right_x, show_demo_idx=2, plot=True)
# </editor-fold>

# <editor-fold desc="7-dim Uni example">
# raw_demo = np.load(files('rofunc.data.LFD_ML').joinpath('LeftHand.npy'))
# raw_demo = np.expand_dims(raw_demo, axis=0)
# demos_x = np.vstack((raw_demo[:, 82:232, :], raw_demo[:, 233:383, :], raw_demo[:, 376:526, :]))
# model, rep = rf.tpgmm.uni(demos_x, show_demo_idx=1, plot=True)
# </editor-fold>


def test_7d_bi_tpgmm():
    # <editor-fold desc="7-dim Bi example">
    left_raw_demo = np.load(files('rofunc.data.LFD_ML').joinpath('LeftHand.npy'))
    right_raw_demo = np.load(files('rofunc.data.LFD_ML').joinpath('RightHand.npy'))
    left_raw_demo = np.expand_dims(left_raw_demo, axis=0)
    right_raw_demo = np.expand_dims(right_raw_demo, axis=0)
    demos_left_x = np.vstack((left_raw_demo[:, 82:232, :], left_raw_demo[:, 233:383, :], left_raw_demo[:, 376:526, :]))
    demos_right_x = np.vstack(
        (right_raw_demo[:, 82:232, :], right_raw_demo[:, 233:383, :], right_raw_demo[:, 376:526, :]))

    model_l, model_r, rep_l, rep_r = rf.tpgmm.bi(demos_left_x, demos_right_x, show_demo_idx=2, plot=False)
    # </editor-fold>


if __name__ == '__main__':
    test_7d_bi_tpgmm()
