import numpy as np
import rofunc as rf
from importlib_resources import files
from isaacgym import gymutil

args = gymutil.parse_arguments(description="Franka Attractor Example")


def franka_run_traj():
    # <editor-fold desc="Run trajectory">
    traj = np.load(files('rofunc.data').joinpath('taichi_1l.npy'))
    rf.franka.run_traj(args, traj, for_test=True)
    # </editor-fold>


# <editor-fold desc="Show the interactive mode">
# rf.franka.show(args)
# </editor-fold>


# if __name__ == '__main__':
#     test_franka_run_traj()
