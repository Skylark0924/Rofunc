import math

import numpy as np

L = 0.25
d = 0.215


def transform_torsoendtoarmbase(data, D):
    R_1 = np.array([[math.cos(data[0]), 0, math.sin(data[0])], [0, 1, 0],
                    [-math.sin(data[0]), 0, math.cos(data[0])]])
    R_2 = np.array([[1, 0, 0], [0, math.cos(data[1]), -math.sin(data[1])],
                    [0, math.sin(data[1]), math.cos(data[1])]])
    R_3 = np.array(
        [[math.cos(data[2]), -math.sin(data[2]), 0], [math.sin(data[2]), math.cos(data[2]), 0],
         [0, 0, 1]])
    R_4 = np.array([[math.cos(data[3]), 0, math.sin(data[3])], [0, 1, 0],
                    [-math.sin(data[3]), 0, math.cos(data[3])]])
    T = np.around(np.array([R_4 @ R_1 @ R_2 @ R_3]).reshape(-1, 3), decimals=6)
    T = np.r_[np.c_[T, D.T], np.array([[0, 0, 0, 1]])]
    return T


def transform_torsobasetotorsoend(theta):
    p = np.empty([3])
    p[0] = L * (math.sin(theta[1]) + math.cos(theta[1] + theta[2])) * math.cos(theta[0])
    p[1] = L * (math.sin(theta[1]) + math.cos(theta[1] + theta[2])) * math.sin(theta[0])
    p[2] = d + L * (math.cos(theta[1]) - math.sin(theta[1] + theta[2]))
    R = np.array(
        [[math.cos(theta[0]), -math.sin(theta[0]), 0], [math.sin(theta[0]), math.cos(theta[0]), 0],
         [0, 0, 1]])
    return p, R


def transform_mobilebasetoarmbase(torso_joint):
    # Fixed Base Transformation
    T_MobileBaseToTorsoBase = np.array([[1, 0, 0, 0.2375], [0, 1, 0, 0], [0, 0, 1, 0.53762], [0, 0, 0, 1]])
    T_TorsoEndToLeftArmBase = transform_torsoendtoarmbase(
        np.array([math.pi / 2, -math.pi / 4, -math.pi / 6, -math.pi / 18]), np.array([[-0.08537, 0.07009, 0.2535]]))
    T_TorsoEndToRightArmBase = transform_torsoendtoarmbase(
        np.array([math.pi / 2, math.pi / 4, math.pi / 6, -math.pi / 18]), np.array([[-0.08537, -0.07009, 0.2535]]))

    # Current Joint States of Torso
    p, R = transform_torsobasetotorsoend(torso_joint)
    T_TorsoBaseToTorsoEnd = np.r_[np.c_[R, p.T], np.array([[0, 0, 0, 1]])]

    # Transformation Matrix from CURI base to Left/Right Arm Base under the torso configuration
    T_MobileBaseToLeftArmBase = T_MobileBaseToTorsoBase @ T_TorsoBaseToTorsoEnd @ T_TorsoEndToLeftArmBase
    # print(T_CURIBaseToLeftArmBase)

    T_MobileBaseToRightArmBase = T_MobileBaseToTorsoBase @ T_TorsoBaseToTorsoEnd @ T_TorsoEndToRightArmBase
    # print(T_CURIBaseToRightArmBase)

    return T_MobileBaseToLeftArmBase, T_MobileBaseToRightArmBase


def transform_armbasetoarmend(arm_joint):
    # create DH parameters given by franka-emika
    DH = np.array(
        [[0, 0, 0.333, arm_joint[0]], [-math.pi / 2, 0, 0, arm_joint[1]], [math.pi / 2, 0, 0.316, arm_joint[2]],
         [math.pi / 2, 0.0825, 0, arm_joint[3]], [-math.pi / 2, -0.0825, 0.384, arm_joint[4]],
         [math.pi / 2, 0, 0, arm_joint[5]], [math.pi / 2, 0.088, 0.107, arm_joint[6]]])
    alpha_franka = DH[:, 0]
    a_franka = DH[:, 1]
    d_franka = DH[:, 2]
    q_franka = DH[:, 3]
    T = np.empty([len(DH), 4, 4])
    for i in range(len(DH)):
        T[i, :, :] = np.array([[math.cos(q_franka[i]), -math.sin(q_franka[i]), 0, a_franka[i]],
                               [math.sin(q_franka[i]) * math.cos(alpha_franka[i]),
                                math.cos(q_franka[i]) * math.cos(alpha_franka[i]), -math.sin(alpha_franka[i]),
                                -math.sin(alpha_franka[i]) * d_franka[i]],
                               [math.sin(q_franka[i]) * math.sin(alpha_franka[i]),
                                math.cos(q_franka[i]) * math.sin(alpha_franka[i]), math.cos(alpha_franka[i]),
                                math.cos(alpha_franka[i]) * d_franka[i]],
                               [0, 0, 0, 1]])
    T_ArmBaseToArmEnd = np.around(T[0] @ T[1] @ T[2] @ T[3] @ T[4] @ T[5] @ T[6], decimals=6)
    return T_ArmBaseToArmEnd  # Arm end represents the flange of Franka Emila


def transform_mobilebasetoarmend(torso_joint, arm_joint_left, arm_joint_right):
    T_MobileBaseToLeftArmBase, T_MobileBaseToRightArmBase = transform_mobilebasetoarmbase(torso_joint)
    T_MobileBaseToLeftArmEnd = T_MobileBaseToLeftArmBase @ transform_armbasetoarmend(arm_joint_left)
    T_MobileBaseToRightArmEnd = T_MobileBaseToRightArmBase @ transform_armbasetoarmend(arm_joint_right)
    return T_MobileBaseToLeftArmEnd, T_MobileBaseToRightArmEnd


def transform_worldbasetomobilebase(data):
    translation = np.array([[data[0], data[1], 0]])
    rotation = np.array([[math.cos(data[2]), -math.sin(data[2]), 0], [math.sin(data[2]), math.cos(data[2]), 0],
                         [0, 0, 1]])
    T_WorldBaseToMobileBase = np.r_[np.c_[rotation, translation.T], np.array([[0, 0, 0, 1]])]
    return T_WorldBaseToMobileBase


def transform_optitrackbasetomobilebase():
    translation = np.array([[0.2195, 0, -1.11462]])
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    T_OptitrackBaseToMobileBase = np.r_[np.c_[rotation, translation.T], np.array([[0, 0, 0, 1]])]
    return T_OptitrackBaseToMobileBase


def transform_optitrackbasetoarmend(torso_joint, arm_joint_left, arm_joint_right):
    T_OptitrackBaseToMobileBase = transform_optitrackbasetomobilebase()
    T_MobileBaseToLeftArmEnd, T_MobileBaseToRightArmEnd = transform_mobilebasetoarmend(torso_joint, arm_joint_left,
                                                                                       arm_joint_right)
    T_OptitrackBaseToLeftArmEnd = T_OptitrackBaseToMobileBase @ T_MobileBaseToLeftArmEnd
    T_OptitrackBaseToRightArmEnd = T_OptitrackBaseToMobileBase @ T_MobileBaseToRightArmEnd
    return T_OptitrackBaseToLeftArmEnd, T_OptitrackBaseToRightArmEnd


if __name__ == '__main__':
    torso_joint = np.array([0, -0.7, 0.1])
    arm_joint_left = np.array([0, 0, 0, 0, 0, 0, 0])
    arm_joint_right = np.array([0, 0, 0, 0, 0, 0, 0])
    # T_MobileBaseToLeftArmEnd, T_MobileBaseToRightArmEnd = transform_mobilebasetoarmend(torso_joint, arm_joint_left, arm_joint_right)
    # print(T_MobileBaseToLeftArmEnd, '\n', T_MobileBaseToRightArmEnd)
    T_l, T_r = transform_optitrackbasetoarmend(torso_joint, arm_joint_left, arm_joint_right)
    print(T_l, '\n', T_r)
