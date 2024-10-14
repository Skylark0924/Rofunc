def get_fk_from_chain(chain, joint_value, export_link_name=None):
    """
    Get the forward kinematics from a serial chain
    :param chain:
    :param joint_value:
    :param export_link_name:
    :return: the pose of the end effector, and the transformation matrices of all links
    """
    # do forward kinematics and get transform objects; end_only=False gives a dictionary of transforms for all links
    ret = chain.forward_kinematics(joint_value)
    # look up the transform for a specific link
    if export_link_name is not None:
        pose = ret[export_link_name]
    else:
        pose = None
    return pose, ret


def get_fk_from_model(model_path: str, joint_value, export_link=None, verbose=False):
    """
    Get the forward kinematics from a URDF or MuJoCo XML file
    :param model_path: the path of the URDF or MuJoCo XML file
    :param joint_value: the value of the joints
    :param export_link: the name of the end effector link
    :param verbose: whether to print the chain
    :return: the pose of the end effector, and the transformation matrices of all links
    """
    from rofunc.utils.robolab.kinematics.pytorch_kinematics_utils import build_chain_from_model

    chain = build_chain_from_model(model_path, verbose)
    pose, ret = get_fk_from_chain(chain, joint_value, export_link)
    return pose, ret