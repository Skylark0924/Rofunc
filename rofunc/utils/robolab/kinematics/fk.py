from urdfpy import URDF


def get_fk_from_model(file_path, joint_name, joint_value, export_link='panda_left_link7'):
    """
    Get the forward kinematics of the robot from the 3D model
    :param file_path:
    :param joint_name:
    :param joint_value:
    :param export_link:
    :return:
    """
    robot = URDF.load(file_path)
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
