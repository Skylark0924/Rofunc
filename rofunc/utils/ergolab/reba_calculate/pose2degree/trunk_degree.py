import numpy as np
import math
import transformations as tf
import rofunc as rf


class TrunkDegree:
    def __init__(self, joints_position,joints_orientation):
        self.joints_position = joints_position
        self.joints_orientation = joints_orientation

    def trunk_plane(self):
        m_body_number = rf.ergolab.body_part_number()
        trunk_joint_numbers = m_body_number.trunk_upper_body()

        # finding a plane of upper body
        u = self.joints_position[trunk_joint_numbers[1]] - self.joints_position[trunk_joint_numbers[0]]
        v = self.joints_position[trunk_joint_numbers[3]] - self.joints_position[trunk_joint_numbers[0]]

        normal_plane = np.cross(u, v)
        return normal_plane

    def trunk_flex_calculator(self):
        normal_plane = self.trunk_plane()
        z_vector = np.array([0, 0, 1])

        trunk_flex = rf.ergolab.get_angle_between_degs(z_vector, normal_plane) - 90
        return trunk_flex

    def trunk_side_calculator(self):
        m_body_number = rf.ergolab.body_part_number()
        trunk_joint_numbers = m_body_number.trunk_upper_body()

        normal_plane_xz = np.array([1, 0, 0])
        z_vector = np.array([0, 0, 1])
        spine_vector = self.joints_position[trunk_joint_numbers[2]] - self.joints_position[trunk_joint_numbers[0]]
        project_spine_on_xz_plane = spine_vector - np.dot(spine_vector, normal_plane_xz) * normal_plane_xz

        trunk_side_bending = rf.ergolab.get_angle_between_degs(z_vector, project_spine_on_xz_plane)

        return trunk_side_bending

    def trunk_twist_calculator(self):
        # In here the rotor needed to transfer orientation frame of core joint to neck joint is calculated
        # this considered as twist
        m_body_number = rf.ergolab.body_part_number()
        trunk_joint_numbers = m_body_number.trunk_upper_body()
        q1 = np.eye(4)
        q2 = np.eye(4)
        q1[:3, :3] = self.joints_orientation[trunk_joint_numbers[2]]# neck
        q2[:3, :3] = self.joints_orientation[trunk_joint_numbers[0]]# core
        q1 = tf.quaternion_from_matrix(q1)
        q2 = tf.quaternion_from_matrix(q2)
        # finding the rotor that express rotation between two orientational frame(between outer and inner joint)
        rotor = rf.ergolab.find_rotation_quaternion(q1, q2)
        trunk_twist = math.acos(abs(rotor[0])) * 2 * (180 / np.pi)
        return trunk_twist

    def trunk_degrees(self):
        trunk_flex_degree = self.trunk_flex_calculator()
        trunk_side_bending_degree = self.trunk_side_calculator()
        trunk_torsion_degree = self.trunk_twist_calculator()

        return [trunk_flex_degree,trunk_side_bending_degree,trunk_torsion_degree]