from rofunc.utils.oslab.path import check_package_exist
from rofunc.utils.logger.beauty_logger import beauty_print

check_package_exist("kinpy")

import kinpy as kp


def build_chain_from_model(model_path: str, verbose=False):
    """
    Build a serial chain from a URDF or MuJoCo XML file
    :param model_path: the path of the URDF or MuJoCo XML file
    :param verbose: whether to print the chain
    :return: robot kinematics chain
    """

    if model_path.endswith(".urdf"):
        chain = kp.build_chain_from_urdf(open(model_path).read())
    elif model_path.endswith(".xml"):
        chain = kp.build_chain_from_mjcf(open(model_path).read())
    else:
        raise ValueError("Invalid model path")

    if verbose:
        beauty_print("Robot chain:")
        print(chain)
        beauty_print("Robot joints:")
        print(chain.get_joint_parameter_names())
    return chain