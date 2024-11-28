import rofunc as rf


class LADegrees:
    def __init__(self, joints_position):
        self.joints_position = joints_position

    def lower_arm_degree(self):
        m_body_number = rf.ergolab.body_part_number()
        right_arm_joint_numbers = m_body_number.right_arm()
        left_arm_joint_numbers = m_body_number.left_arm()

        right_shoulder_elbow_vector = self.joints_position[right_arm_joint_numbers[1]] - self.joints_position[right_arm_joint_numbers[0]]
        left_shoulder_elbow_vector = self.joints_position[left_arm_joint_numbers[1]] - self.joints_position[left_arm_joint_numbers[0]]

        right_elbow_wrist_vector = self.joints_position[right_arm_joint_numbers[2]] - self.joints_position[right_arm_joint_numbers[1]]
        left_elbow_wrist_vector = self.joints_position[left_arm_joint_numbers[2]] - self.joints_position[left_arm_joint_numbers[1]]

        # right and left arm degree in saggital plane
        right_degree = rf.ergolab.get_angle_between_degs(right_shoulder_elbow_vector, right_elbow_wrist_vector)
        left_degree = rf.ergolab.get_angle_between_degs(left_shoulder_elbow_vector, left_elbow_wrist_vector)

        return [right_degree,left_degree]