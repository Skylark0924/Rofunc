"""
CURI Interactive Mode
============================================================

Show the interactive mode of the CURI simulator.
"""

from isaacgym import gymutil

import rofunc as rf

args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

rf.curi.show(args, visual_obs_flag=True)
