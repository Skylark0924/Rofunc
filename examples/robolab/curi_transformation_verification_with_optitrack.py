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
    time = np.reshape(np.arange(0, len(data) / 120, 1 / 120), (-1, 1))
    data = np.hstack((time, data))
    return data


def process_ros_joint_states_csv(path):
    data = np.array(pd.read_csv(path))[:, 11:18]
    time = np.reshape(np.arange(0, len(data) / 480, 1 / 480), (-1, 1))
    data = np.hstack((time, data))[::4]
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
        '/home/lee/Data/20230425_transformation verification with optitrack/csv/robotpose.csv')
    end_pose_left = process_ros_optitrack_csv(
        '/home/lee/Data/20230425_transformation verification with optitrack/csv/leftpose.csv')
    end_pose_right = process_ros_optitrack_csv(
        '/home/lee/Data/20230425_transformation verification with optitrack/csv/rightpose.csv')
    arm_joint_left = process_ros_joint_states_csv(
        '/home/lee/Data/20230425_transformation verification with optitrack/csv/leftjoint.csv')
    arm_joint_right = process_ros_joint_states_csv(
        '/home/lee/Data/20230425_transformation verification with optitrack/csv/rightjoint.csv')
    plt.plot(end_pose_left[:, 2])
    plt.show()

    errors_left = np.zeros([len(robot_pose), 3])
    errors_right = np.zeros([len(robot_pose), 3])
    transform_position_left = np.zeros([len(robot_pose), 3])
    transform_position_right = np.zeros([len(robot_pose), 3])
    for i in range(len(robot_pose)):
        transform_pose_left, transform_pose_right = rf.robolab.transform_optitrackbasetoarmend(np.array([0, -0.7, 0.1]), arm_joint_left[i, 1:], arm_joint_right[i, 1:])
        transform_position_left[i] = transform_pose_left[:3, 3]
        transform_position_right[i] = transform_pose_right[:3, 3]
        errors_left[i], errors_right[i] = compute_errors(arm_joint_left[i, 1:], arm_joint_right[i, 1:],
                                                         robot_pose[i, 1:], end_pose_left[i, 1:], end_pose_right[i, 1:])

    plt.plot(transform_position_left[:, 1])
    plt.show()

    t = np.arange(0, len(robot_pose)/120, 1/120)

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

    plt.savefig('/home/lee/Data/20230425_transformation verification with optitrack/errors.png', dpi=300)
    plt.show()

    # errors_left, errors_right = compute_errors(arm_joint_left[2000, 1:], arm_joint_right[2000, 1:], robot_pose[2000, 1:], end_pose_left[2000, 1:], end_pose_right[2000, 1:])
    # print(errors_left, '\n', errors_right)
