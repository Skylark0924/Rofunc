from __future__ import print_function

import math

from fk_new_config import transform_mobilebasetoarmend, transform_torsoendtoarmbase
import numpy as np
import pinocchio
from numpy.linalg import norm, solve


def transform_torso(torso_joint):
    T_torso = np.empty([len(torso_joint), 4, 4])
    T_torso[0] = np.array([[math.cos(torso_joint[0]), -math.sin(torso_joint[0]), 0, 0.2375],
                          [math.sin(torso_joint[0]), math.cos(torso_joint[0]), 0, 0], [0, 0, 1, 0.75262], [0, 0, 0, 1]])
    T_torso[1] = np.array(
        [[1, 0, 0, 0.25 * math.sin(torso_joint[1])], [0, 1, 0, 0],
         [0, 0, 1, 0.25 * math.cos(torso_joint[1])], [0, 0, 0, 1]])
    T_torso[2] = np.array(
        [[1, 0, 0, 0.25 * math.cos(torso_joint[1] + torso_joint[2])], [0, 1, 0, 0],
         [0, 0, 1, 0.25 * -math.sin(torso_joint[1] + torso_joint[2])], [0, 0, 0, 1]])

    return T_torso


def transform_left_arm(arm_joint):
    T_TorsoEndToLeftArmBase = transform_torsoendtoarmbase(
        np.array([math.pi / 2, -math.pi / 4, -math.pi / 6, -math.pi / 18]), np.array([[-0.08537, 0.07009, 0.2535]]))
    DH = np.array(
        [[0, 0, 0.333, arm_joint[0]], [-math.pi / 2, 0, 0, arm_joint[1]], [math.pi / 2, 0, 0.316, arm_joint[2]],
         [math.pi / 2, 0.0825, 0, arm_joint[3]], [-math.pi / 2, -0.0825, 0.384, arm_joint[4]],
         [math.pi / 2, 0, 0, arm_joint[5]], [math.pi / 2, 0.088, 0.107, arm_joint[6]]])
    alpha_franka = DH[:, 0]
    a_franka = DH[:, 1]
    d_franka = DH[:, 2]
    q_franka = DH[:, 3]
    T_arm = np.empty([len(DH), 4, 4])
    for i in range(len(DH)):
        T_arm[i, :, :] = np.array([[math.cos(q_franka[i]), -math.sin(q_franka[i]), 0, a_franka[i]],
                               [math.sin(q_franka[i]) * math.cos(alpha_franka[i]),
                                math.cos(q_franka[i]) * math.cos(alpha_franka[i]), -math.sin(alpha_franka[i]),
                                -math.sin(alpha_franka[i]) * d_franka[i]],
                               [math.sin(q_franka[i]) * math.sin(alpha_franka[i]),
                                math.cos(q_franka[i]) * math.sin(alpha_franka[i]), math.cos(alpha_franka[i]),
                                math.cos(alpha_franka[i]) * d_franka[i]],
                               [0, 0, 0, 1]])
    T_arm[0] = T_TorsoEndToLeftArmBase @ T_arm[0]

    return T_arm


def compute_jacobian(T):
    p = np.zeros([len(T) + 1, 3])
    z = np.empty([len(T), 3])
    T_new = np.zeros([len(T), 4, 4])
    J = np.empty([len(T), 6])
    for i in range(len(T)):
        T_new[i] = np.eye(4)
        for j in range(i + 1):
            T_new[i] = T_new[i] @ T[j]
        p[i + 1] = (T_new[i, 0:3, 0:3] @ p[0].T + T_new[i, 0:3, 3]).T
        z[i] = T_new[i, 0:3, 2]
    for i in range(len(T)):
        J[i] = np.append(np.cross(z[i], (p[len(T)] - p[i])), z[i])
    return J


def ik_new_config(pose, ori, joint_id):
    q = np.array([0, -0.7, 0.1, 0, 0, 0, 0, 0, 0, 0])   #initial joint states
    oMdes = pinocchio.SE3(ori, pose)
    eps = 1e-4
    IT_MAX = 1000
    DT = 1e-1
    damp = 1e-12

    i = 0
    while True:
        oMi_l, oMi_r = transform_mobilebasetoarmend(q[0:3], q[3:10], q[3:10])
        T_overall = np.append(transform_torso(q[0:3]), transform_left_arm(q[3:10]), axis=0)
        oMi = pinocchio.SE3(oMi_l[0:3, 0:3], oMi_l[0:3, 3])
        dMi = oMdes.actInv(oMi)
        err = pinocchio.log(dMi).vector
        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        J = compute_jacobian(T_overall[:joint_id]).T
        v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q = q + v * DT
        if not i % 10:
            print('%d: error = %s' % (i, err.T))
        i += 1

    if success:
        print("Convergence achieved!")
    else:
        print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

    print('\nresult: %s' % q.flatten().tolist())
    print('\nfinal error: %s' % err.T)
    return q


if __name__ == '__main__':
    pose = np.array([1, 0, 1])
    ori = np.eye(3)
    q = ik_new_config(pose, ori, joint_id=10)