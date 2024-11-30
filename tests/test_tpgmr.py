import numpy as np

import rofunc as rf


def test_2d_uni_tpgmr():
    demo_points = np.array([[[0, 0], [-1, 8], [4, 3], [2, 1], [4, 3]],
                            [[0, -2], [-1, 7], [3, 2.5], [2, 1.6], [4, 3]],
                            [[0, -1], [-1, 8], [4, 5.2], [2, 1.1], [4, 3.5]]])
    demos_x = rf.data_generator.multi_bezier_demos(demo_points)  # (3, 50, 2): 3 demos, each has 50 points

    start_xdx = [demos_x[i][0] for i in range(len(demos_x))]  # TODO: change to xdx
    end_xdx = [demos_x[i][-1] for i in range(len(demos_x))]
    task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
    Repr = rf.RofuncML.TPGMR(demos_x, task_params, plot=False)
    model = Repr.fit()

    traj, _ = Repr.reproduce(model, show_demo_idx=2)

#
# def test_7d_uni_tpgmr():
#     # Uni_3d
#     raw_demo = np.load('../examples/data/LFD_ML/LeftHand.npy')
#     demos_x = [raw_demo[500:635, :], raw_demo[635:770, :], raw_demo[770:905, :]]
#
#     # TP-GMR
#     start_xdx = [demos_x[i][0] for i in range(len(demos_x))]  # TODO: change to xdx
#     end_xdx = [demos_x[i][-1] for i in range(len(demos_x))]
#     task_params = {'frame_origins': [start_xdx, end_xdx], 'frame_names': ['start', 'end']}
#     Repr = rf.RofuncML.TPGMR(demos_x, task_params, plot=False)
#     model = Repr.fit()
#
#     # Reproductions for the same situations
#     traj, _ = Repr.reproduce(model, show_demo_idx=2)


if __name__ == '__main__':
    test_2d_uni_tpgmr()
    # test_7d_uni_tpgmr()
