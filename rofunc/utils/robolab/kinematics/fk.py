import ikpy.chain
import ikpy.utils.plot as plot_utils
import numpy as np
import math

def fk(urdf, joint_state, export_joint="EE"):
    export_joint_pose = ...
    return export_joint_pose


if __name__ == '__main__':
    my_robot = ikpy.chain.Chain.from_urdf_file("/home/ubuntu/Rofunc/rofunc/simulator/assets/urdf/franka_description/robots/franka_panda.urdf")
    print(my_robot)



    # urdf = ...
    # joint_state = []
    # pose = fk(urdf, joint_state)
    # print(pose)
