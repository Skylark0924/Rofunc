import numpy as np
from urdfpy import URDF
from .utils import check_urdf


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

    export_pose = forward_kinematics[robot.links[link_name.index(export_link)]]

    return robot, export_pose, cfg


# ['reference', 'summit_xls_front_right_wheel_joint', 'summit_xls_front_left_wheel_joint', 'summit_xls_back_left_wheel_joint',
# 'summit_xls_back_right_wheel_joint', 'torso_actuated_joint1', 'torso_actuated_joint2', 'torso_actuated_joint3',
# 'head_actuated_joint1', 'panda_left_joint1', 'panda_right_joint1', 'head_actuated_joint2', 'panda_left_joint2',
# 'panda_right_joint2', 'panda_right_joint3', 'panda_left_joint3', 'panda_left_joint4', 'panda_right_joint4',
# 'panda_left_joint5', 'panda_right_joint5', 'panda_right_joint6', 'panda_left_joint6', 'panda_left_joint7', 'panda_right_joint7',
# 'panda_left_finger_joint1', 'panda_right_finger_joint1']

if __name__ == '__main__':
    urdf_path = '/home/lee/Rofunc/rofunc/simulator/assets/urdf/curi/urdf/curi.urdf'
    actuated_joint_name = check_urdf(urdf_path)
    joint_value = [0.0, 1.0, 1.0, 1.0, 1.0, 2.1605735253736429e-16, 0.5326418108438093, -0.35713090036667317, 0.0,
                   -0.5248189935729404, 0.5248189935729395, 0.0, 1.1210417041164564, 1.1210417041164547,
                   1.9946571243973779, -1.9946571243973743, 0.729107826387619, 0.7291078263876194, 0.22015132006660315,
                   -0.2201513200666057, -0.1688895250088166, -0.1688895250088167, -0.14403010737637342,
                   0.14403010737637464, 0.0, 0.0]
    joint_value = np.array(joint_value)
    robot, export_pose, cfg = fk(urdf_path, actuated_joint_name, joint_value)
    print(export_pose)
    robot.show(cfg=cfg)

    # # Joint trajectory visualization
    # joint_range = np.array([0.0, -np.pi / 4] * len(actuated_joint_name)).reshape((len(actuated_joint_name), 2))
    # cfg_trajectory = {}
    # for key, value in zip(actuated_joint_name, joint_range):
    #     if key != 'reference':
    #         cfg_trajectory[key] = value
    # robot.animate(cfg_trajectory=cfg_trajectory)
