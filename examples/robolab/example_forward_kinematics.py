"""
FK from models
========================

Forward kinematics from URDF or MuJoCo XML files.
"""

import pprint

import math

import rofunc as rf

# forward kinematics from URDF or MuJoCo XML files with python functions
rf.logger.beauty_print("########## Forward kinematics from URDF or MuJoCo XML files with python functions ##########")
rf.logger.beauty_print("---------- Forward kinematics for Franka Panda using URDF file ----------")
model_path = "/home/ubuntu/Github/Xianova_Robotics/Rofunc-secret/rofunc/simulator/assets/urdf/franka_description/robots/franka_panda.urdf"
joint_value = {"panda_joint1": 0.0,
               "panda_joint2": -math.pi / 4.0,
               "panda_joint3": 0.0,
               "panda_joint4": math.pi / 2.0,
               "panda_joint5": 0.0,
               "panda_joint6": math.pi / 4.0,
               "panda_joint7": 0.0,
               "panda_finger_joint1": 0.0,
               "panda_finger_joint2": 0.0}
export_link = "panda_hand"
pos, rot, tg = rf.robolab.get_fk_from_model(model_path, joint_value, export_link, verbose=True)
rf.logger.beauty_print(f"Position of {export_link}: {pos}")
rf.logger.beauty_print(f"Rotation of {export_link}: {rot}")
pprint.pprint(tg, width=1)

rf.logger.beauty_print("---------- Forward kinematics for Bruce using MJCF file ----------")
model_path = "/home/ubuntu/Github/Xianova_Robotics/Rofunc-secret/rofunc/simulator/assets/mjcf/bruce/bruce.xml"
joint_value = {"hip_yaw_r": 0.0,
               "hip_roll_r": 0.0,
               "hip_pitch_r": 0.0,
               "knee_pitch_r": 0.0,
               "ankle_pitch_r": 0.0,
               "hip_yaw_l": 0.0,
               "hip_roll_l": 0.0,
               "hip_pitch_l": 0.0,
               "knee_pitch_l": 0.0,
               "ankle_pitch_l": 0.0,
               "shoulder_pitch_r": 0.0,
               "shoulder_roll_r": 0.0,
               "elbow_pitch_r": 0.0,
               "shoulder_pitch_l": 0.0,
               "shoulder_roll_l": 0.0,
               "elbow_pitch_l": 0.0}

export_link = "elbow_pitch_link_r"

pos, rot, tg = rf.robolab.get_fk_from_model(model_path, joint_value, export_link, verbose=True)
rf.logger.beauty_print(f"Position of {export_link}: {pos}")
rf.logger.beauty_print(f"Rotation of {export_link}: {rot}")
pprint.pprint(tg, width=1)

rf.logger.beauty_print("########## Forward kinematics from URDF or MuJoCo XML files with RobotModel class ##########")
rf.logger.beauty_print("---------- Forward kinematics for Franka Panda using URDF file ----------")
model_path = "/home/ubuntu/Github/Xianova_Robotics/Rofunc-secret/rofunc/simulator/assets/urdf/franka_description/robots/franka_panda.urdf"
joint_value = {"panda_joint1": 0.0,
               "panda_joint2": -math.pi / 4.0,
               "panda_joint3": 0.0,
               "panda_joint4": math.pi / 2.0,
               "panda_joint5": 0.0,
               "panda_joint6": math.pi / 4.0,
               "panda_joint7": 0.0,
               "panda_finger_joint1": 0.0,
               "panda_finger_joint2": 0.0}
export_link = "panda_hand"
robot = rf.robolab.RobotModel(model_path, verbose=True)
pos, rot, tg = robot.get_fk(joint_value, export_link)
rf.logger.beauty_print(f"Position of {export_link}: {pos}")
rf.logger.beauty_print(f"Rotation of {export_link}: {rot}")
pprint.pprint(tg, width=1)
