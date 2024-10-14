"""
Jacobian from models
========================

Jacobian from URDF or MuJoCo XML files.
"""

import rofunc as rf

rf.logger.beauty_print("########## Jacobian from URDF or MuJoCo XML files with RobotModel class ##########")
model_path = "../../rofunc/simulator/assets/mjcf/bruce/bruce.xml"
joint_value = [0.1 for _ in range(16)]

export_link = "elbow_pitch_link_r"

# # Build the robot model with pytorch_kinematics, kinpy is not supported for MJCF files
robot = rf.robolab.RobotModel(model_path, solve_engine="pytorch_kinematics", verbose=True)

# Get the jacobian of export_link
J = robot.get_jacobian(joint_value, export_link)
print(J)

# Get the jacobian at a point offset from the export_link
point = [0.1, 0.1, 0.1]
J = robot.get_jacobian(joint_value, export_link, locations=point)
print(J)
