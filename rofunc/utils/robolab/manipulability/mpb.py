import numpy as np

from rofunc.utils.robolab.kinematics.jacobian import get_jacobian_from_model
from rofunc.utils.visualab.ellipsoid import ellipsoid_plot3d


def get_mpb_from_model(file_path, end_link_name, joint_values, show=False):
    """
    Get the manipulability of the robot from the 3D model
    :param file_path: path to the robot model file
    :return: the manipulability of the robot
    """
    J = get_jacobian_from_model(file_path, end_link_name, joint_values)

    # calculate the velocity manipulability
    A = J @ J.T
    vel_eig_values, vel_eig_vectors = np.linalg.eig(A)
    vel_M = np.sqrt(np.linalg.det(A))

    # calculate the force manipulability
    force_eig_values = np.reciprocal(vel_eig_values)
    force_eig_vectors = vel_eig_vectors
    A_inv = np.linalg.inv(A)
    # force_eig_values, force_eig_vectors = np.linalg.eig(A_inv)
    force_M = np.sqrt(np.linalg.det(A_inv))

    if show:
        ellipsoid_plot3d(np.array([vel_eig_values[:3], force_eig_values[:3]]), mode='given',
                         Rs=np.array([vel_eig_vectors[:3, :3], vel_eig_vectors[:3, :3]]))

    return vel_M, vel_eig_values, vel_eig_vectors, force_M, force_eig_values, force_eig_vectors


if __name__ == '__main__':
    urdf_path = "/home/ubuntu/Github/Manipulation/kinpy/examples/kuka_iiwa/model.urdf"
    joint_values = [0, -np.pi / 4.0, -np.pi / 4.0, np.pi / 2.0, 0.0, np.pi / 4.0, -np.pi / 4.0]
    get_mpb_from_model(urdf_path, end_link_name='lbr_iiwa_link_7', joint_values=joint_values, show=True)
