from rofunc.utils.robolab.kinematics.mjcf import build_chain_from_mjcf


def build_chain_from_model(model_path: str, verbose=False):
    """
    Build a serial chain from a URDF or MuJoCo XML file

    :param model_path: the path of the URDF or MuJoCo XML file
    :param verbose: whether to print the chain
    :return: robot kinematics chain
    """
    import pytorch_kinematics as pk

    if model_path.endswith(".urdf"):
        chain = pk.build_chain_from_urdf(open(model_path).read())
    elif model_path.endswith(".xml"):
        chain = build_chain_from_mjcf(model_path)
    else:
        raise ValueError("Invalid model path")

    if verbose:
        print(chain)
    return chain
