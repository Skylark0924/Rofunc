import rofunc as rf
import numpy as np
import os


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

def test_7d_uni_tpgmm():
    raw_demo = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LFD_ML/LeftHand.npy'))
    demos_x = [raw_demo[500:635, :], raw_demo[635:770, :], raw_demo[770:905, :]]

    representation = rf.lfd.tpgmm.TPGMM(demos_x, plot=False)
    model = representation.fit()

    # Reproductions for the same situations
    traj = representation.reproduce(model, show_demo_idx=2)

    # Reproductions for new situations
    ref_demo_idx = 2
    A, b = representation.demos_A_xdx[ref_demo_idx][0], representation.demos_b_xdx[ref_demo_idx][0]
    b[1] = b[0]
    task_params = {'A': A, 'b': b}
    traj = representation.generate(model, ref_demo_idx, task_params)


def test_7d_bi_tpgmm():
    left_raw_demo = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LFD_ML/LeftHand.npy'))
    right_raw_demo = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LFD_ML/RightHand.npy'))
    demos_left_x = [left_raw_demo[500:635, :], left_raw_demo[635:770, :], left_raw_demo[770:905, :]]
    demos_right_x = [right_raw_demo[500:635, :], right_raw_demo[635:770, :], right_raw_demo[770:905, :]]

    representation = rf.lfd.tpgmm.TPGMMBi(demos_left_x, demos_right_x, plot=False)
    model_l, model_r = representation.fit()

    # Reproductions for the same situations
    traj_l, traj_r = representation.reproduce(model_l, model_r, show_demo_idx=2)

    # Reproductions for new situations
    ref_demo_idx = 2
    A_l, b_l = representation.repr_l.demos_A_xdx[ref_demo_idx][0], representation.repr_l.demos_b_xdx[ref_demo_idx][0]
    b_l[1] = b_l[0]
    A_r, b_r = representation.repr_r.demos_A_xdx[ref_demo_idx][0], representation.repr_r.demos_b_xdx[ref_demo_idx][0]
    b_r[1] = b_r[0]
    task_params = {'Left': {'A': A_l, 'b': b_l}, 'Right': {'A': A_r, 'b': b_r}}
    traj_l, traj_r = representation.generate(model_l, model_r, ref_demo_idx, task_params)


if __name__ == '__main__':
    test_7d_uni_tpgmm()
    test_7d_bi_tpgmm()
