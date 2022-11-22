import numpy as np
from urdfpy import URDF
from utils import check_urdf


def fk(urdf_path, joint_name, joint_value, export_link='panda_left_link7'):
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

#['reference', 'summit_xls_front_right_wheel_joint', 'summit_xls_front_left_wheel_joint', 'summit_xls_back_left_wheel_joint',
# 'summit_xls_back_right_wheel_joint', 'torso_actuated_joint1', 'torso_actuated_joint2', 'torso_actuated_joint3',
# 'head_actuated_joint1', 'panda_left_joint1', 'panda_right_joint1', 'head_actuated_joint2', 'panda_left_joint2',
# 'panda_right_joint2', 'panda_right_joint3', 'panda_left_joint3', 'panda_left_joint4', 'panda_right_joint4',
# 'panda_left_joint5', 'panda_right_joint5', 'panda_right_joint6', 'panda_left_joint6', 'panda_left_joint7', 'panda_right_joint7',
# 'panda_left_finger_joint1', 'panda_right_finger_joint1']

if __name__ == '__main__':
    urdf_path = '/home/lee/Rofunc/rofunc/simulator/assets/urdf/curi/urdf/curi.urdf'
    actuated_joint_name = check_urdf(urdf_path)

    # joint_value = [0.2] * len(actuated_joint_name)
    joint_value = [0] * len(actuated_joint_name)
    joint_value[5] = -1.1755335500416668
    joint_value[9] = 0.35518036639555406
    joint_value[12] = 1.4328213040329258
    joint_value[15] = -0.7650259308960912
    joint_value[16] = 1.6815538882284287
    joint_value[18] = -1.203551423269827
    joint_value[21] = 0.6320274890381881
    joint_value = np.array(joint_value)
    robot, export_pose = fk(urdf_path, actuated_joint_name, joint_value)

    joint_range = np.array([0.0, -np.pi / 4] * len(actuated_joint_name)).reshape((len(actuated_joint_name), 2))
    cfg_trajectory = {}
    for key, value in zip(actuated_joint_name, joint_range):
        if key != 'reference':
            cfg_trajectory[key] = value
    robot.animate(cfg_trajectory=cfg_trajectory)
