"""
Visualize robots and objects
============================================================

This example shows how to visualize robots and objects in the Isaac Gym simulator in an interactive viewer.
"""

from isaacgym import gymutil

import rofunc as rf

args = gymutil.parse_arguments()
args.use_gpu_pipeline = False

# CURI
# CURIsim = rf.sim.CURISim(args)
# CURIsim.init()
# CURIsim.show(visual_obs_flag=True)

# walker
# walkersim = rf.sim.WalkerSim(args)
# walkersim.show()

# CURI-mini
# CURIminisim = rf.sim.CURIminiSim(args)
# CURIminisim.show()

# franka
# frankasim = rf.sim.FrankaSim(args)
# frankasim.show()

# baxter
# baxtersim = rf.sim.BaxterSim(args)
# baxtersim.show()

# sawyer
# sawyersim = rf.sim.SawyerSim(args)
# sawyersim.show()

# gluon
# Gluonsim = rf.sim.GluonSim(args, init_pose_vec=[1, 0, 0])
# Gluonsim.show(visual_obs_flag=False)

# Multi Robots
CURIsim = rf.sim.CURISim(args)
CURIsim2 = rf.sim.CURISim(args, init_pose_vec=[2, 0, 0, 0., 0.707107, 0.707107, 0.])

MRsim = rf.sim.MultiRobotSim(args, robot_sims={"CURI": CURIsim, "CURI2": CURIsim2})
MRsim.init()
MRsim.show()
