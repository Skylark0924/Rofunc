def get_fk_from_model(urdf_path, joint_name, joint_value, export_link):
    """

    :param urdf_path:
    :param joint_name:
    :param joint_value:
    :param export_link:
    :return:
    """
    from urdfpy import URDF

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
