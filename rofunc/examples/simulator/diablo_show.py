"""
CURI mini Interactive Mode
============================================================

Show the interactive mode of the CURI mini simulator.
"""

from isaacgym import gymutil

import rofunc as rf

args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

rf.curi_mini.show(args)
