"""
IK from models
========================

Inverse kinematics from URDF or MuJoCo XML files.
"""

import rofunc as rf

rf.logger.beauty_print("########## Inverse kinematics from URDF or MuJoCo XML files with RobotModel class ##########")
rf.logger.beauty_print("---------- Inverse kinematics for Franka Panda using URDF file ----------")
model_path = "../../rofunc/simulator/assets/urdf/franka_description/robots/franka_panda.urdf"

ee_pose = [0, 0, 0, 0, 0, 0, 1]
export_link = "panda_hand"
robot = rf.robolab.RobotModel(model_path, solve_engine="pytorch_kinematics", verbose=True)

# Get ik solution
ret = robot.get_ik(ee_pose, export_link)
print(ret.solutions)

# Get ik solution near the current configuration
cur_configs = [[-1.7613,  2.7469, -3.5611, -3.8847,  2.7940,  1.9055,  1.9879]]
ret = robot.get_ik(ee_pose, export_link, cur_configs=cur_configs)
print(ret.solutions)
