import rofunc as rf


def check_urdf(urdf_path):
    from urdfpy import URDF

    robot = URDF.load(urdf_path)
    # link_number = len(robot.links)
    # joint_number = len(robot.joints)

    link_name = []
    for link in robot.links:
        link_name.append(link.name)
    rf.logger.beauty_print('Robot link names (total {}): {}'.format(len(robot.links), link_name), type='info')

    joint_name = []
    for joint in robot.joints:
        joint_name.append(joint.name)
    rf.logger.beauty_print('Robot joint names (total {}): {}'.format(len(robot.joints), joint_name), type='info')

    actuated_joint_name = []
    for joint in robot.actuated_joints:
        actuated_joint_name.append(joint.name)
    rf.logger.beauty_print(
        'Robot actuated joint names (total {}): {}'.format(len(robot.actuated_joints), actuated_joint_name),
        type='info')

    # for joint in robot.joints:
    #     print('{} connects {} to {}'.format(
    #         joint.name, joint.parent, joint.child))

    return link_name, joint_name, actuated_joint_name
