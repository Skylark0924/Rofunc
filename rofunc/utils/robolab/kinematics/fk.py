# import ikpy.chain
# import ikpy.utils.plot as plot_utils
# import numpy as np
# import math
from urdfpy import URDF

def fk(urdf, joint_state, export_joint="EE"):
    export_joint_pose = ...
    return export_joint_pose


if __name__ == '__main__':
    # my_robot = ikpy.chain.Chain.from_urdf_file("/home/ubuntu/Rofunc/rofunc/simulator/assets/urdf/franka_description/robots/franka_panda.urdf")
    my_robot = URDF.load('/home/lee/Rofunc/rofunc/simulator/assets/urdf/curi/urdf/curi.urdf')

    # for link in my_robot.links:
    #     print(link.name)

    # for joint in my_robot.joints:
    #     print(joint.name)
    #
    # for joint in my_robot.joints:
    #     print('{} connects {} to {}'.format(
    #     joint.name, joint.parent, joint.child
    #     ))

    for joint in my_robot.actuated_joints:
        print(joint.name)

    """Add the following code after line 287 in 'mesh.py' of package 'urdfpy' to run the visualization:
    uv = np.zeros((3 * len(mesh.faces), 2), float)
    """
    my_robot.show(cfg={
        'panda_left_joint1': 0.0,
        'panda_left_joint2': 0.0,
        'panda_right_joint1': 0.0,
        'panda_right_joint2': 0.0
    })

    # urdf = ...
    # joint_state = []
    # pose = fk(urdf, joint_state)
    # print(pose)
