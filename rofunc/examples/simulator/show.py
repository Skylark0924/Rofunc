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
CURIsim = rf.curi.CURISim(args)
CURIsim.show(visual_obs_flag=True)

# walker
# walkersim = rf.simulator.RobotSim(args, robot_name="walker")
# walkersim.show()

# CURI-mini
# CURIminisim = rf.simulator.RobotSim(args, robot_name="CURI-mini")
# CURIminisim.show()

# franka
# frankasim = rf.simulator.RobotSim(args, robot_name="franka", fix_base_link=True)
# frankasim.show()

# baxter
# baxtersim = rf.simulator.RobotSim(args, robot_name="baxter")
# baxtersim.show()

# sawyer
# sawyersim = rf.simulator.RobotSim(args, robot_name="sawyer")
# sawyersim.show()
