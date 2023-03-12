import rofunc as rf
import numpy as np
import os

# Uni
# demo_points = np.array([[[0, 0], [-1, 8], [4, 3], [2, 1], [4, 3]],
#                         [[0, -2], [-1, 7], [3, 2.5], [2, 1.6], [4, 3]],
#                         [[0, -1], [-1, 8], [4, 5.2], [2, 1.1], [4, 3.5]]])
# demos_x = rf.data_generator.multi_bezier_demos(demo_points)  # (3, 50, 2): 3 demos, each has 50 points
# start_pose, end_pose = [-1, -2], [6, 6]
# model, rep = rf.tpgmr.uni(demos_x, 2, start_pose, end_pose, plot=True)


def test_7d_uni_tpgmr():
    # Uni_3d
    raw_demo = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LFD_ML/LeftHand.npy'))
    demos_x = [raw_demo[500:635, :], raw_demo[635:770, :], raw_demo[770:905, :]]

    # TP-GMR
    representation = rf.lfd.tpgmr.TPGMR(demos_x, plot=False)
    model = representation.fit()

    # Reproductions for the same situations
    traj = representation.reproduce(model, show_demo_idx=2)


if __name__ == '__main__':
    test_7d_uni_tpgmr()
