import rofunc as rf
import numpy as np
from rofunc.config.get_config import *


def test_2d_com_ilqr():
    cfg = get_config('./', 'ilqr_com')

    Mu = np.asarray([3.5, 4])  # Target
    MuCoM = np.asarray([.4, 0])

    u0 = np.zeros(cfg.nbVarU * (cfg.nbData - 1))  # Initial control command
    a = .7
    x0 = np.array([np.pi / 2 - a, 2 * a, - a, np.pi - np.pi / 4, 3 * np.pi / 4])  # Initial state (in joint space)

    rf.lqr.uni_com(Mu, MuCoM, u0, x0, cfg, for_test=True)


if __name__ == '__main__':
    test_2d_com_ilqr()
