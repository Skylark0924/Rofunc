import rofunc as rf
import numpy as np
import os
from rofunc.config.utils import get_config

cfg = get_config('./planning', 'lqt_cp_dmp')
cfg.nbDeriv = 3


# <editor-fold desc="3d example">
# TODO: not work
# data_raw = np.load('/home/ubuntu/Data/2022_09_09_Taichi/rep3_r.npy')
# # filter_indices = [i for i in range(0, len(data_raw) - 10, 5)]
# # filter_indices.append(len(data_raw) - 1)
# MuPos = data_raw  # pose
# MuVel = np.gradient(MuPos)[0] / cfg.dt
# MuAcc = np.gradient(MuVel)[0] / cfg.dt
# via_points = np.hstack((MuPos, MuVel, MuAcc)).T
# </editor-fold>


def test_2d_cp_dmp_lqt():
    # <editor-fold desc="2d letter example data">
    from scipy.interpolate import interp1d

    x = np.load(os.path.join(rf.utils.get_rofunc_path(), 'data/LQT_LQR/S.npy'))[0, :, :2].T

    f_pos = interp1d(np.linspace(0, np.size(x, 1) - 1, np.size(x, 1), dtype=int), x, kind='cubic')
    MuPos = f_pos(np.linspace(0, np.size(x, 1) - 1, cfg.nbData))  # Position
    MuVel = np.gradient(MuPos)[1] / cfg.dt
    MuAcc = np.gradient(MuVel)[1] / cfg.dt
    # Position, velocity and acceleration profiles as references
    via_points = np.vstack((MuPos, MuVel, MuAcc))
    # </editor-fold>

    cfg.nbData = len(via_points[0])
    rf.lqt.uni_cp_dmp(via_points, cfg, for_test=True)


if __name__ == '__main__':
    test_2d_cp_dmp_lqt()
