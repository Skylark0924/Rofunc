import rofunc as rf
import numpy as np
from rofunc.config.get_config import *


def test_2d_bi_ilqr():
    cfg = get_config('./', 'ilqr_bi')

    Mu = np.array([[-1, -1.5, 4, 2]]).T  # Target
    MuCoM = np.array([0, 1.4])

    u0 = np.zeros(cfg.nbVarU * (cfg.nbData - 1))  # Initial control command
    x0 = np.array([np.pi / 2, np.pi / 2, np.pi / 3, -np.pi / 2, -np.pi / 3])  # Initial pose

    rf.lqr.uni_bi(Mu, MuCoM, u0, x0, cfg, for_test=True)


if __name__ == '__main__':
    test_2d_bi_ilqr()
