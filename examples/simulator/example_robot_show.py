"""
Visualize robots and objects
============================================================

This example shows how to visualize robots and objects in the Isaac Gym simulator in an interactive viewer.
"""

import isaacgym
import rofunc as rf

# # CURI
args = rf.config.get_sim_config("CURI")
CURIsim = rf.sim.CURISim(args)
CURIsim.show(visual_obs_flag=False)

# walker
# args = rf.config.get_sim_config("Walker")
# walkersim = rf.sim.WalkerSim(args)
# walkersim.show()

# CURI-mini
# args = rf.config.get_sim_config("CURImini")
# CURIminisim = rf.sim.RobotSim(args)
# CURIminisim.show(visual_obs_flag=True)

# franka
# args = rf.config.get_sim_config("Franka")
# frankasim = rf.sim.FrankaSim(args)
# frankasim.show()

# baxter
# args = rf.config.get_sim_config("Baxter")
# baxtersim = rf.sim.RobotSim(args)
# baxtersim.show()

# sawyer
# args = rf.config.get_sim_config("Sawyer")
# sawyersim = rf.sim.RobotSim(args)
# sawyersim.show()

# gluon
# args = rf.config.get_sim_config("Gluon")
# Gluonsim = rf.sim.GluonSim(args)
# Gluonsim.show()

# # qbsofthand
# args = rf.config.get_sim_config("QbSoftHand")
# QbSoftHandsim = rf.sim.QbSoftHandSim(args)
# QbSoftHandsim.show()

# HOTU
# args = rf.config.get_sim_config("Humanoid")
# Humanoidsim = rf.sim.HumanoidSim(args)
# Humanoidsim.show()

# Unitree H1
# args = rf.config.get_sim_config("UnitreeH1")
# UnitreeH1sim = rf.sim.RobotSim(args)
# UnitreeH1sim.show()

# Bruce humanoid
# args = rf.config.get_sim_config("Bruce")
# Brucesim = rf.sim.RobotSim(args)
# Brucesim.show()

# ZJU humanoid
# args = rf.config.get_sim_config("ZJUHumanoid")
# ZJUHumanoidsim = rf.sim.RobotSim(args)
# ZJUHumanoidsim.show()

# Themis
# args = rf.config.get_sim_config("Themis")
# Themissim = rf.sim.RobotSim(args)
# Themissim.show()

# TODO: Multi Robots
# curi_args = rf.config.get_sim_config("CURI")
# CURIsim = rf.sim.CURISim(curi_args)
# walker_args = rf.config.get_sim_config("Walker")
# walkersim = rf.sim.WalkerSim(walker_args)
#
# MRsim = rf.sim.MultiRobotSim(args, robot_sims={"CURI": CURIsim, "walker": walkersim})
# MRsim.show()
