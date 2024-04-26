"""
LLM control a robot with pre-defined low-level motion API
============================================================

This example allows you to control a robot with LLM and pre-defined low-level motion API.
"""

import isaacgym
import numpy as np

import rofunc as rf

# traj_l = np.load('../data/LQT_LQR/taichi_1l.npy')
# traj_l[:, 0] += 0.2
# traj_l[:, 1] = -traj_l[:, 1]
# traj_l[:, 3:] = [1, 0, 0, 0]
# # target_pose = traj_l[10]
# traj_l = traj_l[::20]

args = rf.config.get_sim_config("CURI_LLM")
CURIsim = rf.sim.CURISim(args)
# CURIsim.run_hand_reach_target_pose(target_pose=[traj_l],
#                                    attracted_hand=["panda_left_hand"])
CURIsim.run_with_text_commands()
