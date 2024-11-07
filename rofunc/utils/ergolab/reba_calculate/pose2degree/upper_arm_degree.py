import numpy as np
import rofunc as rf


class UpperArmDegree:
    def __init__(self, joints_position):
        self.joints_position = joints_position

    def trunk_plane(self):
        m_body_number = rf.ergolab.body_part_number()
        trunk_joint_numbers = m_body_number.trunk_upper_body()

        # finding a plane of upper body
        u = self.joints_position[trunk_joint_numbers[1]] - self.joints_position[trunk_joint_numbers[0]]
        v = self.joints_position[trunk_joint_numbers[3]] - self.joints_position[trunk_joint_numbers[0]]

        normal_plane = np.cross(u, v)
        return normal_plane

    def upper_arm_flex(self):
        m_body_number = rf.ergolab.body_part_number()
        right_upper_arm_joint_numbers = m_body_number.right_arm()
        left_upper_arm_joint_numbers = m_body_number.left_arm()
        trunk_joint_numbers = m_body_number.trunk_whole_body()

        right_upper_arm_vector = self.joints_position[right_upper_arm_joint_numbers[1]] - self.joints_position[
            right_upper_arm_joint_numbers[0]]

        left_upper_arm_vector = self.joints_position[left_upper_arm_joint_numbers[1]] - self.joints_position[
            left_upper_arm_joint_numbers[0]]

        normal_trunk_plane = self.trunk_plane()
        spine_vector = self.joints_position[trunk_joint_numbers[3]] - self.joints_position[trunk_joint_numbers[2]]

        sagittal_plane_normal = np.cross(normal_trunk_plane, spine_vector)
        sagittal_plane_normal /= np.linalg.norm(sagittal_plane_normal)

        right_arm_projection = right_upper_arm_vector - np.dot(right_upper_arm_vector,
                                                               sagittal_plane_normal) * sagittal_plane_normal
        left_arm_projection = left_upper_arm_vector - np.dot(left_upper_arm_vector,
                                                             sagittal_plane_normal) * sagittal_plane_normal

        # Calculate flexion angles using the projection and the normalized spine vector
        flex_right_upper_arm = np.degrees(np.arccos(np.dot(right_arm_projection, spine_vector) /
                                                    (np.linalg.norm(right_arm_projection) * np.linalg.norm(
                                                        spine_vector))))
        flex_left_upper_arm = np.degrees(np.arccos(np.dot(left_arm_projection, spine_vector) /
                                                   (np.linalg.norm(left_arm_projection) * np.linalg.norm(
                                                       spine_vector))))

        return [flex_right_upper_arm, flex_left_upper_arm]

    def upper_arm_abduct(self):
        m_body_number = rf.ergolab.body_part_number()
        right_upper_arm_joint_numbers = m_body_number.right_arm()
        left_upper_arm_joint_numbers = m_body_number.left_arm()
        trunk_joint_numbers = m_body_number.trunk_whole_body()

        right_upper_arm_vector = self.joints_position[right_upper_arm_joint_numbers[1]] - self.joints_position[
            right_upper_arm_joint_numbers[0]]
        left_upper_arm_vector = self.joints_position[left_upper_arm_joint_numbers[1]] - self.joints_position[
            left_upper_arm_joint_numbers[0]]

        normal_trunk_plane = self.trunk_plane()

        proj_right_upperarm_on_plane = right_upper_arm_vector - np.dot(right_upper_arm_vector,
                                                                       normal_trunk_plane) * normal_trunk_plane

        proj_left_upperarm_on_plane = left_upper_arm_vector - np.dot(left_upper_arm_vector,
                                                                     normal_trunk_plane) * normal_trunk_plane

        spine_vector = self.joints_position[trunk_joint_numbers[3]] - self.joints_position[trunk_joint_numbers[2]]

        right_side_degree = rf.ergolab.get_angle_between_degs(spine_vector, proj_right_upperarm_on_plane)

        left_side_degree = rf.ergolab.get_angle_between_degs(spine_vector, proj_left_upperarm_on_plane)

        if np.dot(np.cross(spine_vector, right_upper_arm_vector), normal_trunk_plane) < 0:
            # if the arm go to the body: adduction
            right_side_degree *= -1

        if np.dot(np.cross(spine_vector, left_upper_arm_vector), normal_trunk_plane) > 0:
            left_side_degree *= -1

        return [right_side_degree, left_side_degree]

    def shoulder_rise(self):
        m_body_number = rf.ergolab.body_part_number()
        trunk_joint_numbers = m_body_number.trunk_upper_body()
        right_shoulder_joint_numbers = m_body_number.right_shoulder()
        left_shoulder_joint_numbers = m_body_number.left_shoulder()
        spine_vector = self.joints_position[trunk_joint_numbers[0]] - self.joints_position[trunk_joint_numbers[2]]
        right_shoulder_vector = self.joints_position[right_shoulder_joint_numbers[1]] - self.joints_position[
            right_shoulder_joint_numbers[0]]
        left_shoulder_vector = self.joints_position[left_shoulder_joint_numbers[1]] - self.joints_position[left_shoulder_joint_numbers[0]]

        right_shoulder_rise_degree = 90 - rf.ergolab.get_angle_between_degs(spine_vector, right_shoulder_vector)
        left_shoulder_rise_degree = 90 - rf.ergolab.get_angle_between_degs(spine_vector, left_shoulder_vector)

        return [right_shoulder_rise_degree, left_shoulder_rise_degree]

    def upper_arm_degrees(self):
        flexion = self.upper_arm_flex()
        side = self.upper_arm_abduct()
        shoulder_rise = self.shoulder_rise()

        right_flexion = flexion[0]
        left_flexion = flexion[1]

        right_side = side[0]
        left_side = side[1]

        right_shoulder_rise = shoulder_rise[0]
        left_shoulder_rise = shoulder_rise[1]

        return [right_flexion,left_flexion,right_side,left_side,right_shoulder_rise,left_shoulder_rise]