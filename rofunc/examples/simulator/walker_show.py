"""
Ubtech Walker Interactive Mode
============================================================

Show the interactive mode of the Ubtech Walker simulator.
"""

from isaacgym import gymutil

import rofunc as rf

args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

rf.walker.show(args)
