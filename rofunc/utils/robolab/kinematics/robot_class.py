from rofunc.utils.robolab.kinematics.fk import get_fk_from_chain
from rofunc.utils.robolab.coord import convert_ori_format, convert_quat_order, homo_matrix_from_quat_tensor


class RobotModel:
    def __init__(self, model_path: str, solve_engine: str = "pytorch_kinematics", device="cpu", verbose=False):
        """
        Initialize a robot model from a URDF or MuJoCo XML file

        :param model_path: the path of the URDF or MuJoCo XML file
        :param solve_engine: the engine to solve the kinematics, ["pytorch_kinematics", "kinpy", "all"]
        :param device: the device to run the computation
        :param verbose: whether to print the chain
        """
        assert solve_engine in ["pytorch_kinematics", "kinpy"], "Unsupported solve engine."
        self.solve_engine = solve_engine
        self.device = device

        if self.solve_engine == "pytorch_kinematics":
            from rofunc.utils.robolab.kinematics import pytorch_kinematics_utils as pk_utils
            self.chain = pk_utils.build_chain_from_model(model_path, verbose)
        elif self.solve_engine == "kinpy":
            from rofunc.utils.robolab.kinematics import kinpy_utils as kp_utils
            self.chain = kp_utils.build_chain_from_model(model_path, verbose)

    def get_fk(self, joint_value, export_link_name):
        """
        Get the forward kinematics from a chain

        :param joint_value:
        :param export_link_name:
        :return: the position, rotation of the end effector, and the transformation matrices of all links
        """
        if self.solve_engine == "pytorch_kinematics":
            pose, ret = get_fk_from_chain(self.chain, joint_value, export_link_name)
            m = pose.get_matrix()
            pos = m[:, :3, 3]
            rot = convert_ori_format(m[:, :3, :3], "mat", "quat")
            return pos, rot, ret
        elif self.solve_engine == "kinpy":
            pose, ret = get_fk_from_chain(self.chain, joint_value, export_link_name)
            pos = pose.pos
            rot = pose.rot
            rot = convert_quat_order(rot, "wxyz", "xyzw")
            return pos, rot, ret

    def get_ik(self, ee_pose, export_link_name):
        """
        Get the inverse kinematics from a chain

        :param ee_pose: the pose of the end effector, 7D vector with the first 3 elements as position and the last 4 elements as rotation
        :param export_link_name:
        :return: the joint values
        """
        if self.solve_engine == "pytorch_kinematics":
            return get_ik_from_chain(self.chain, ee_pose[:3], ee_pose[3:], self.device)
        elif self.solve_engine == "kinpy":
            import kinpy as kp
            self.serial_chain = kp.chain.SerialChain(self.chain, export_link_name)
            homo_matrix = homo_matrix_from_quat_tensor(ee_pose[3:], ee_pose[:3])
            return self.serial_chain.inverse_kinematics(homo_matrix)
