"""
Construct custom human model from Xsens data
=============================================

This example shows how to construct a custom human model (URDF) from Xsens data.
"""

from isaacgym import gymutil

import rofunc as rf

args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

# CURI
CURIsim = rf.sim.CURISim(args)
CURIsim.show(visual_obs_flag=True)
