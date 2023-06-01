import rofunc as rf
import numpy as np
import os
from rofunc.config.utils import get_config


# <editor-fold desc="3d example">
# TODO: not work
# data_raw = np.load('/home/ubuntu/Data/2022_09_09_Taichi/rep3_r.npy')
# # filter_indices = [i for i in range(0, len(data_raw) - 10, 5)]
# # filter_indices.append(len(data_raw) - 1)
# MuPos = data_raw  # pose
# MuVel = np.gradient(MuPos)[0] / cfg.dt
# MuAcc = np.gradient(MuVel)[0] / cfg.dt
# via_points = np.hstack((MuPos, MuVel, MuAcc)).T
# state_noise = np.hstack(
#     (-1, -.2, 1, 0, 0, 0, 0, np.zeros(cfg.nbVarX - cfg.nbVarPos))).reshape(
#     (cfg.nbVarX, 1))  # Simulated noise on 3d state
# </editor-fold>


def test_2d_cp_dmp_lqt():
    from scipy.interpolate import interp1d

    cfg = get_config('./planning', 'lqt_cp_dmp')
    cfg.nbDeriv = 3

    x = np.load('../examples/data/LQT_LQR/S.npy')[0, :, :2].T

    f_pos = interp1d(np.linspace(0, np.size(x, 1) - 1, np.size(x, 1), dtype=int), x, kind='cubic')
    MuPos = f_pos(np.linspace(0, np.size(x, 1) - 1, cfg.nbData))  # Position
    MuVel = np.gradient(MuPos)[1] / cfg.dt
    MuAcc = np.gradient(MuVel)[1] / cfg.dt
    # Position, velocity and acceleration profiles as references
    via_points = np.vstack((MuPos, MuVel, MuAcc))

    state_noise = np.vstack(
        (np.array([[3], [-0.5]]), np.zeros((cfg.nbVarX - cfg.nbVarU, 1))))  # Simulated noise on 2d state

    cfg.nbData = len(via_points[0])
    controller = rf.planning_control.lqt.LQTCPDMP(via_points, cfg)
    u_hat, x_hat = controller.solve(state_noise, for_test=True)


if __name__ == '__main__':
    test_2d_cp_dmp_lqt()
