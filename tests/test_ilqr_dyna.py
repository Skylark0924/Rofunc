import rofunc as rf
import numpy as np
from rofunc.config.get_config import *


def test_2d_dyna_ilqr():
    cfg = get_config('./planning', 'ilqr')

    # via-points
    Mu = np.array([[2, 1, -np.pi / 3], [3, 2, -np.pi / 3]])  # Via-points [x, y, orientation]
    Rot = np.zeros([cfg.nbPoints, 2, 2])  # Object orientation matrices

    # Object rotation matrices
    for t in range(cfg.nbPoints):
        orn_t = Mu[t, -1]
        Rot[t, :, :] = np.asarray([
            [np.cos(orn_t), -np.sin(orn_t)],
            [np.sin(orn_t), np.cos(orn_t)]
        ])

    u0 = np.zeros(cfg.nbVarU * (cfg.nbData - 1))  # Initial control command
    x0 = np.array([3 * np.pi / 4, -np.pi / 2, -np.pi / 4])  # Initial state
    v0 = np.array([0, 0, 0])  # initial velocity (in joint space)

    rf.lqr.uni_dyna(Mu, Rot, u0, x0, v0, cfg, for_test=True)


if __name__ == '__main__':
    test_2d_dyna_ilqr()
