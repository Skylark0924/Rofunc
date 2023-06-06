"""
CURI Interactive Mode
============================================================

Show the interactive mode of the CURI simulator.
"""

from isaacgym import gymutil

import rofunc as rf

args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

object_name = "Cabinet"
rf.object.show(args, object_name)
