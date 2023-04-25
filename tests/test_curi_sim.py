import os

import numpy as np
from isaacgym import gymutil
import rofunc as rf

args = gymutil.parse_arguments()


def curi_run_traj():
    # <editor-fold desc="Run trajectory">
    traj_l = np.load(os.path.join(rf.file.get_rofunc_path(), 'data/taichi_1l.npy'))
    traj_r = np.load(os.path.join(rf.file.get_rofunc_path(), 'data/taichi_1r.npy'))
    rf.lqt.plot_3d_bi(traj_l, traj_r, ori=False, for_test=True)

    CURIsim = rf.sim.CURISim(args)
    CURIsim.run_traj(traj=[traj_l, traj_r], update_freq=0.001)
    # </editor-fold>

# <editor-fold desc="Show the interactive mode">
# rf.curi.show(args)
# </editor-fold>


# if __name__ == '__main__':
#     curi_run_traj()
