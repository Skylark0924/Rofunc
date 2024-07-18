from rofunc.utils.robolab.kinematics.utils import build_chain_from_model


def get_fk_from_chain(chain, joint_value, export_link_name):
    """
    Get the forward kinematics from a serial chain

    :param chain:
    :param joint_value:
    :param export_link_name:
    :return: the position, rotation of the end effector, and the transformation matrices of all links
    """
    import pytorch_kinematics as pk

    # do forward kinematics and get transform objects; end_only=False gives a dictionary of transforms for all links
    ret = chain.forward_kinematics(joint_value)
    # look up the transform for a specific link
    tg = ret[export_link_name]
    # get transform matrix (1,4,4), then convert to separate position and unit quaternion
    m = tg.get_matrix()
    pos = m[:, :3, 3]
    rot = pk.matrix_to_quaternion(m[:, :3, :3])
    return pos, rot, ret


def get_fk_from_model(model_path: str, joint_value, export_link, verbose=False):
    """
    Get the forward kinematics from a URDF or MuJoCo XML file

    :param model_path: the path of the URDF or MuJoCo XML file
    :param joint_value: the value of the joints
    :param export_link: the name of the end effector link
    :param verbose: whether to print the chain
    :return: the position, rotation of the end effector, and the transformation matrices of all links
    """

    chain = build_chain_from_model(model_path, verbose)
    pos, rot, tg = get_fk_from_chain(chain, joint_value, export_link)
    return pos, rot, tg
