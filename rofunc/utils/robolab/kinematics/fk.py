import numpy as np
from urdfpy import URDF
from utils import check_urdf


def fk(urdf_path, joint_name, joint_value, export_link='panda_left_hand'):
    robot = URDF.load(urdf_path)
    link_name = []
    for link in robot.links:
        str = link.name
        link_name.append(str)

    cfg = {}
    for key, value in zip(joint_name, joint_value):
        if key != 'reference':
            cfg[key] = value
    forward_kinematics = robot.link_fk(cfg=cfg)

    # robot.show(cfg=cfg)

    export_pose = forward_kinematics[robot.links[link_name.index(export_link)]]

    return robot, export_pose


if __name__ == '__main__':
    urdf_path = '/home/lee/Rofunc/rofunc/simulator/assets/urdf/curi/urdf/curi.urdf'
    actuated_joint_name = check_urdf(urdf_path)

    joint_value = [0.2] * len(actuated_joint_name)
    robot, export_pose = fk(urdf_path, actuated_joint_name, joint_value)

    joint_range = np.array([0.0, -np.pi / 4] * len(actuated_joint_name)).reshape((len(actuated_joint_name), 2))
    cfg_trajectory = {}
    for key, value in zip(actuated_joint_name, joint_range):
        if key != 'reference':
            cfg_trajectory[key] = value
    robot.animate(cfg_trajectory=cfg_trajectory)
