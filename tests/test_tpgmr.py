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
    raw_demo = np.expand_dims(raw_demo, axis=0)
    demos_x = np.vstack((raw_demo[:, 82:232, :], raw_demo[:, 233:383, :], raw_demo[:, 376:526, :]))
    show_demo_idx = 2
    start_pose = demos_x[show_demo_idx][-1]
    end_pose = demos_x[show_demo_idx][0]
    model, rep = rf.tpgmr.uni(demos_x, show_demo_idx, start_pose, end_pose, plot=False)


if __name__ == '__main__':
    test_7d_uni_tpgmr()
