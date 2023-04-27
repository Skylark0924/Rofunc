"""
CURI FK transformation verification with optitrack
========================

Verify the precision of transformation matrix by optitrack
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rofunc as rf


def process_ros_optitrack_csv(path):
    data = np.array(pd.read_csv(path))[:, 4:]
    time = np.reshape(np.array(pd.read_csv(path))[:, 0], (-1, 1)) * 1e-11
    time = time - np.int_(time)
    data = np.hstack((time, data))
    return data


def process_ros_joint_states_csv(path):
    data = np.array(pd.read_csv(path))[:, 11:18]
    time = np.reshape(np.array(pd.read_csv(path))[:, 0], (-1, 1)) * 1e-11
    time = time - np.int_(time)
    data = np.hstack((time, data))
    return data


def compute_errors(arm_joint_left, arm_joint_right, robot_pose, end_pose_left, end_pose_right):
    optitrack_position_left = end_pose_left[:3] - robot_pose[:3]
    # optitrack_position_left[0], optitrack_position_left[1] = optitrack_position_left[1], optitrack_position_left[0]
    optitrack_position_right = end_pose_right[:3] - robot_pose[:3]
    # optitrack_position_right[0], optitrack_position_right[1] = optitrack_position_right[1], optitrack_position_right[0]
    transform_pose_left, transform_pose_right = rf.robolab.transform_optitrackbasetoarmend(np.array([0, -0.7, 0.1]),
                                                                                           arm_joint_left,
                                                                                           arm_joint_right)
    transform_position_left = transform_pose_left[:3, 3]
    transform_position_right = transform_pose_right[:3, 3]
    errors_left = optitrack_position_left - transform_position_left
    errors_right = optitrack_position_right - transform_position_right
    return errors_left, errors_right


if __name__ == '__main__':
    robot_pose = process_ros_optitrack_csv(
        '/home/ubuntu/Data/optitrack test/csv_20230426/robotpose.csv')
    end_pose_left = process_ros_optitrack_csv(
        '/home/ubuntu/Data/optitrack test/csv_20230426/leftpose.csv')
    end_pose_right = process_ros_optitrack_csv(
        '/home/ubuntu/Data/optitrack test/csv_20230426/leftpose.csv')
    arm_joint_left = process_ros_joint_states_csv(
        '/home/ubuntu/Data/optitrack test/csv_20230426/leftjoint.csv')
    arm_joint_right = process_ros_joint_states_csv(
        '/home/ubuntu/Data/optitrack test/csv_20230426/leftjoint.csv')

    index = np.zeros(len(robot_pose))
    for i in range(len(robot_pose)):
        arr = arm_joint_left[:, 0]
        value = end_pose_left[i, 0]
        difference_array = np.absolute(arr - value)
        index[i] = difference_array.argmin()
    arm_joint_left = np.array([arm_joint_left[int(i)] for i in index])
    arm_joint_right = arm_joint_left

    t_1 = np.arange(0, 20, 20 / len(end_pose_left))
    plt.plot(t_1, (end_pose_left[:, 1] - robot_pose[:, 1]), label='optitrack')

    errors_left = np.zeros([len(robot_pose), 3])
    errors_right = np.zeros([len(robot_pose), 3])
    transform_position_left = np.zeros([len(arm_joint_left), 3])
    transform_position_right = np.zeros([len(arm_joint_left), 3])
    for i in range(len(arm_joint_left)):
        transform_pose_left, transform_pose_right = rf.robolab.transform_optitrackbasetoarmend(np.array([0, -0.7, 0.1]),
                                                                                               arm_joint_left[i, 1:],
                                                                                               arm_joint_right[i, 1:])
        transform_position_left[i] = transform_pose_left[:3, 3]
        transform_position_right[i] = transform_pose_right[:3, 3]
    for i in range(len(robot_pose)):
        errors_left[i], errors_right[i] = compute_errors(arm_joint_left[i, 1:], arm_joint_right[i, 1:],
                                                         robot_pose[i, 1:], end_pose_left[i, 1:], end_pose_right[i, 1:])
    t_2 = np.arange(0, 20, 20/len(arm_joint_left))
    plt.plot(t_2, transform_position_left[:, 0], label='transform')
    plt.legend(loc="upper right", prop={'size': 12})
    plt.show()

    t = np.arange(0, len(robot_pose)/120, 1/120)

    errors = np.zeros(len(errors_left))
    for i in range(len(errors_left)):
        errors[i] = np.sqrt(errors_left[i, 0] ** 2 + errors_left[i, 1] ** 2 + errors_left[i, 2] ** 2)
    plt.plot(t, errors)
    # plt.savefig('/home/ubuntu/Data/optitrack test/csv_20230426/errors_1.png', dpi=300)
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.subplot(3, 2, 1)
    # plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
    plt.ylabel('(x/m)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.plot(t, errors_left[:, 0], color="royalblue", label='left_arm', linewidth=1.5)
    plt.legend(loc="upper right", prop={'size': 12})

    plt.subplot(3, 2, 2)
    # plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
    # plt.ylabel('m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.plot(t, errors_right[:, 0], color="royalblue", label='right_arm', linewidth=1.5)
    plt.legend(loc="upper right", prop={'size': 12})

    plt.subplot(3, 2, 3)
    # plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
    plt.ylabel('y/m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.plot(t, errors_left[:, 1], color="royalblue", label='left_arm', linewidth=1.5)
    # plt.legend(loc="upper right", prop={'size': 12})

    plt.subplot(3, 2, 4)
    # plt.xlabel('s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
    # plt.ylabel('(m)', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.plot(t, errors_right[:, 1], color="royalblue", label='right_arm', linewidth=1.5)
    # plt.legend(loc="upper right", prop={'size': 12})

    plt.subplot(3, 2, 5)
    plt.xlabel('t/s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
    plt.ylabel('z/m', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.plot(t, errors_left[:, 2], color="royalblue", label='left_arm', linewidth=1.5)
    # plt.legend(loc="upper right", prop={'size': 12})

    plt.subplot(3, 2, 6)
    plt.xlabel('t/s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
    # plt.ylabel('m/s', fontweight="bold", fontdict={'family': 'Times New Roman'}, fontsize=18)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.plot(t, errors_right[:, 2], color="royalblue", label='right_arm', linewidth=1.5)
    # plt.legend(loc="upper right", prop={'size': 12})

    # plt.savefig('/home/ubuntu/Data/optitrack test/csv_20230426/errors_2.png', dpi=300)
    plt.show()

