"""
Construct custom human model from Xsens data
=============================================

This example shows how to construct a custom human model (URDF) from Xsens data.
"""

from isaacgym import gymutil

import rofunc as rf

args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

mvnx_path = '/home/ubuntu/Downloads/SQ_100kg.mvnx'
save_dir = '/home/ubuntu/Data/xsens_mvnx'
mvnx_file = rf.sim.xsens2urdf(mvnx_path, save_dir, human_mass=7, human_height=1.8)

# CURI
human_sim = rf.sim.HumanSim(args, asset_root=save_dir, asset_file='Athlete_0412.urdf')
human_sim.init()
# human_sim.show()
human_sim.run_demo(mvnx_file)
