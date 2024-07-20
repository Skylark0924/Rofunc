"""
IK from models
========================

Inverse kinematics from URDF or MuJoCo XML files.
"""

import pprint

import math
import torch
import rofunc as rf

rf.logger.beauty_print("########## Inverse kinematics from URDF or MuJoCo XML files with RobotModel class ##########")
rf.logger.beauty_print("---------- Inverse kinematics for Franka Panda using URDF file ----------")
model_path = "/home/ubuntu/Github/Xianova_Robotics/Rofunc-secret/rofunc/simulator/assets/urdf/franka_description/robots/franka_panda.urdf"

ee_pose = [0, 0, 0, 0, 0, 0, 1]
export_link = "panda_hand_frame"
robot = rf.robolab.RobotModel(model_path, solve_engine="kinpy", verbose=True)
ret = robot.get_ik(ee_pose, export_link)
print(ret)
