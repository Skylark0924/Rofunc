import numpy as np


def get_jacobian_from_model(file_path, end_link_name, joint_values):
    """
    Get the Jacobian matrix J of the robot from the 3D model, \dot x= J \dot q
    :param file_path:
    :param joint_values:
    :return:
    """
    import kinpy as kp
    if file_path.endswith('gluon.urdf'):
        data = open(file_path)
        xslt_content = data.read().encode()
        chain = kp.build_serial_chain_from_urdf(xslt_content, end_link_name=end_link_name)
    elif file_path.endswith('.urdf'):
        chain = kp.build_serial_chain_from_urdf(open(file_path).read(), end_link_name=end_link_name)
    elif file_path.endswith('.xml'):
        chain = kp.build_serial_chain_from_mjcf(open(file_path).read(), end_link_name=end_link_name)

    J = chain.jacobian(joint_values)

    ret = chain.forward_kinematics(joint_values, end_only=False)

    # viz = kp.Visualizer()
    # viz.add_robot(ret, chain.visuals_map(), mesh_file_path="/home/ubuntu/Github/Manipulation/kinpy/examples/kuka_iiwa/", axes=True)
    # viz.spin()

    return J


if __name__ == '__main__':
    urdf_path = "/home/hengyi/GitHub/Rofunc/rofunc/simulator/assets/urdf/gluon/gluon.urdf"
    joint_values = [0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.0, np.pi / 4.0]
    get_jacobian_from_model(urdf_path, end_link_name='6_Link', joint_values=joint_values)
