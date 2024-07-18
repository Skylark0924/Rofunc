from rofunc.utils.robolab.kinematics.utils import build_chain_from_model
from rofunc.utils.robolab.kinematics.fk import get_fk_from_chain


class RobotModel:
    def __init__(self, model_path: str, verbose=False):
        """
        Initialize a robot model from a URDF or MuJoCo XML file

        :param model_path: the path of the URDF or MuJoCo XML file
        :param verbose: whether to print the chain
        """
        self.chain = build_chain_from_model(model_path, verbose)

    def get_fk(self, joint_value, export_link_name):
        """
        Get the forward kinematics from a serial chain

        :param joint_value:
        :param export_link_name:
        :return: the position, rotation of the end effector, and the transformation matrices of all links
        """
        return get_fk_from_chain(self.chain, joint_value, export_link_name)
