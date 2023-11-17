"""
Visualize robots and objects
============================================================

This example shows how to visualize robots and objects in the Isaac Gym simulator in an interactive viewer.
"""

import rofunc as rf

# CURI
# args = rf.config.get_sim_config("CURI")
# CURIsim = rf.sim.CURISim(args)
# CURIsim.show()

# walker
# args = rf.config.get_sim_config("Walker")
# walkersim = rf.sim.WalkerSim(args)
# walkersim.show()

# CURI-mini
# args = rf.config.get_sim_config("CURImini")
# CURIminisim = rf.sim.CURIminiSim(args)
# CURIminisim.show()

# franka
# args = rf.config.get_sim_config("Franka")
# frankasim = rf.sim.FrankaSim(args)
# frankasim.show()

# baxter
# args = rf.config.get_sim_config("Baxter")
# baxtersim = rf.sim.BaxterSim(args)
# baxtersim.show()

# sawyer
# args = rf.config.get_sim_config("Sawyer")
# sawyersim = rf.sim.SawyerSim(args)
# sawyersim.show()

# gluon
# args = rf.config.get_sim_config("Gluon")
# Gluonsim = rf.sim.GluonSim(args)
# Gluonsim.show()

# # qbsofthand
# args = rf.config.get_sim_config("QbSoftHand")
# QbSoftHandsim = rf.sim.QbSoftHandSim(args)
# QbSoftHandsim.show()

# Humanoid
args = rf.config.get_sim_config("Humanoid")
Humanoidsim = rf.sim.HumanoidSim(args)
Humanoidsim.show()

# TODO: Multi Robots
# curi_args = rf.config.get_sim_config("CURI")
# CURIsim = rf.sim.CURISim(curi_args)
# walker_args = rf.config.get_sim_config("Walker")
# walkersim = rf.sim.WalkerSim(walker_args)
#
# MRsim = rf.sim.MultiRobotSim(args, robot_sims={"CURI": CURIsim, "walker": walkersim})
# MRsim.show()
