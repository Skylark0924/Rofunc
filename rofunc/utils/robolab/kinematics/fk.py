import numpy as np
from urdfpy import URDF
from utils import check_urdf

def fk(urdf_path, joint_name, joint_value):
    robot = URDF.load(urdf_path)
    link_number = len(robot.links)
    cfg = {}
    for key, value in zip(joint_name, joint_value):
        if key != 'reference':
            cfg[key] = value
    forward_kinematics = robot.link_fk(cfg=cfg)

    panda_left_hand = forward_kinematics[robot.links[link_number - 6]]
    panda_right_hand = forward_kinematics[robot.links[link_number - 3]]

    return robot, panda_left_hand, panda_right_hand


if __name__ == '__main__':
    # my_robot = URDF.load('/home/ubuntu/Rofunc/rofunc/simulator/assets/urdf/curi/urdf/curi.urdf')
    urdf_path = '/home/ubuntu/Rofunc/rofunc/simulator/assets/urdf/curi/urdf/curi.urdf'
    actuated_joint_name = check_urdf(urdf_path)
    joint_value = [0] * len(actuated_joint_name)
    joint_range = np.zeros((len(actuated_joint_name), 2))
    joint_range[:, 1] = -np.pi / 4
    # joint_range = [0.0, -np.pi / 4] * len(actuated_joint_name)
    robot, panda_left_hand, panda_right_hand = fk(urdf_path, actuated_joint_name, joint_value)

    # my_robot.show(cfg={
    #     'panda_left_joint1': 2.0,
    #     'panda_left_joint2': 2.0,
    #     'panda_right_joint1': 2.0,
    #     'panda_right_joint2': 2.0
    # })
    cfg_trajectory = {}
    for key, value in zip(actuated_joint_name, joint_range):
        if key != 'reference':
            cfg_trajectory[key] = value
    robot.animate(cfg_trajectory=cfg_trajectory)
    # robot.animate(cfg_trajectory={
    # 'torso_actuated_joint1': [-np.pi / 4, np.pi / 4],
    # 'torso_actuated_joint2': [0.0, -np.pi / 4],
    # 'torso_actuated_joint3': [0.0, np.pi / 4]
    # })

