"""
FD from models
========================

Forward dynamics from URDF or MuJoCo XML files.
"""

import pprint
import math

import rofunc as rf

rf.logger.beauty_print("########## Forward kinematics from URDF or MuJoCo XML files with RobotModel class ##########")
rf.logger.beauty_print("---------- Forward kinematics for Franka Panda using URDF file ----------")
model_path = "../../rofunc/simulator/assets/urdf/franka_description/robots/franka_panda.urdf"

joint_value = [[0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0, 0.0, 0.0],
               [0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0, 0.0, 0.0]]
export_link = "panda_hand"

# # Build the robot model with kinpy
# # Deprecated: kinpy is not supported anymore, just for checking the results!!!! Please use pytorch_kinematics instead.
# robot = rf.robolab.RobotModel(model_path, solve_engine="kinpy", verbose=False)
# # Show the robot chain and joint names, can also be done by verbose=True
# robot.show_chain()
# # Get the forward kinematics of export_link
# pos, rot, ret = robot.get_fk(joint_value, export_link)
#
# # Convert the orientation representation and print the results
# rot = rf.robolab.convert_quat_order(rot, "wxyz", "xyzw")
# rf.logger.beauty_print(f"Position of {export_link}: {pos}")
# rf.logger.beauty_print(f"Rotation of {export_link}: {rot}")
# pprint.pprint(ret, width=1)

# Try the same thing with pytorch_kinematics
robot = rf.robolab.RobotModel(model_path, solve_engine="pytorch_kinematics", verbose=False)
pos, rot, ret = robot.get_fk(joint_value, export_link)
rf.logger.beauty_print(f"Position of {export_link}: {pos}")
rf.logger.beauty_print(f"Rotation of {export_link}: {rot}")
pprint.pprint(ret, width=1)

# rf.logger.beauty_print("---------- Forward kinematics for Bruce Humanoid Robot using MJCF file ----------")
model_path = "../../rofunc/simulator/assets/mjcf/bruce/bruce.xml"
joint_value = [0.0 for _ in range(16)]

export_link = "elbow_pitch_link_r"

# # Build the robot model with pytorch_kinematics, kinpy is not supported for MJCF files
robot = rf.robolab.RobotModel(model_path, solve_engine="pytorch_kinematics", verbose=True)
# Get the forward kinematics of export_link
pos, rot, ret = robot.get_fk(joint_value, export_link)
#
# # Print the results
# rf.logger.beauty_print(f"Position of {export_link}: {pos}")
# rf.logger.beauty_print(f"Rotation of {export_link}: {rot}")
# pprint.pprint(ret, width=1)


model_path = "../../rofunc/simulator/assets/mjcf/hotu/hotu_humanoid.xml"
joint_value = [0.1 for _ in range(34)]

export_link = "left_hand_link_2"

# # Build the robot model with pytorch_kinematics, kinpy is not supported for MJCF files
robot = rf.robolab.RobotModel(model_path, solve_engine="pytorch_kinematics", verbose=True)
# Get the forward kinematics of export_link
pos, rot, ret = robot.get_fk(joint_value, export_link)

# # Print the results
rf.logger.beauty_print(f"Position of {export_link}: {pos}")
rf.logger.beauty_print(f"Rotation of {export_link}: {rot}")
pprint.pprint(ret, width=1)
