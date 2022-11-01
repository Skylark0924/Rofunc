from urdfpy import URDF


def check_urdf(urdf_path):
    robot = URDF.load(urdf_path)
    link_number = len(robot.links)
    joint_number = len(robot.joints)

    link_name = []
    for link in robot.links:
        str = link.name
        link_name.append(str)
    print(link_name)

    joint_name = []
    for joint in robot.joints:
        str = joint.name
        joint_name.append(str)

    actuated_joint_name = []
    for joint in robot.actuated_joints:
        str = joint.name
        actuated_joint_name.append(str)

    # for joint in robot.joints:
    #     print('{} connects {} to {}'.format(
    #         joint.name, joint.parent, joint.child))

    return actuated_joint_name


if __name__ == '__main__':
    urdf_path = '/home/lee/Rofunc/rofunc/simulator/assets/urdf/curi/urdf/curi.urdf'
    actuated_joint_name = check_urdf(urdf_path)
    # print(actuated_joint_name)
