import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utils

def ik(urdf, pose, export_joint="EE"):
    export_joint_state = ...
    return export_joint_state


if __name__ == '__main__':
    robot = ikpy.chain.Chain.from_urdf_file("/home/lee/Rofunc/rofunc/simulator/assets/urdf/curi/urdf/curi.urdf")
    target_position = [0.1, 0.1, 0.5]
    print("The angles of each joints are : ", robot.inverse_kinematics(target_position))
    # urdf = ...
    # pose = []
    # joint_state = ik(urdf, pose)
    # print(joint_state)
