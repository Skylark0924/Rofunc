import numpy as np
import isaacgym
from isaacgym import gymutil
import rofunc as rf
from importlib_resources import files

args = gymutil.parse_arguments()


def curi_run_traj():
    # <editor-fold desc="Run trajectory">
    traj_l = np.load(files('rofunc.data').joinpath('taichi_1l.npy'))
    traj_r = np.load(files('rofunc.data').joinpath('taichi_1r.npy'))
    rf.lqt.plot_3d_bi(traj_l, traj_r, ori=False, for_test=True)
    rf.curi.run_traj_bi(args, traj_l, traj_r, update_freq=0.001, for_test=True)
    # </editor-fold>


# <editor-fold desc="Show the interactive mode">
# rf.curi.show(args)
# </editor-fold>


# if __name__ == '__main__':
#     curi_run_traj()
