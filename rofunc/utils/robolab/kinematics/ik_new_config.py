from __future__ import print_function

import math

from fk_new_config import transform_mobilebasetoarmend, transform_torsoendtoarmbase
import numpy as np
import pinocchio
from numpy.linalg import norm, solve


def transform_torso(torso_joint):
    T_1 = np.array([[math.cos(torso_joint[0]), -math.sin(torso_joint[0]), 0, 0.2375],
                          [math.sin(torso_joint[0]), math.cos(torso_joint[0]), 0, 0], [0, 0, 1, 0.75262], [0, 0, 0, 1]])
    T_2 = np.array(
        [[1, 0, 0, 0.25 * math.sin(torso_joint[1])], [0, 1, 0, 0],
         [0, 0, 1, 0.25 * math.cos(torso_joint[1])], [0, 0, 0, 1]])
    T_3 = np.array(
        [[1, 0, 0, 0.25 * math.cos(torso_joint[1] + torso_joint[2])], [0, 1, 0, 0],
         [0, 0, 1, 0.25 * -math.sin(torso_joint[1] + torso_joint[2])], [0, 0, 0, 1]])

    return T_1, T_2, T_3


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
    T_4 = T_TorsoEndToLeftArmBase @ T[0]
    T_5 = T[1]
    T_6 = T[2]
    T_7 = T[3]
    T_8 = T[4]
    T_9 = T[5]
    T_10 = T[6]

    return T_4, T_5, T_6, T_7, T_8, T_9, T_10


T_1, T_2, T_3 = transform_torso(np.array([0, -0.7, 0.1]))
T_4, T_5, T_6, T_7, T_8, T_9, T_10 = transform_left_arm(np.array([0, 0, 0, 0, 0, 0, 0]))

T_test = T_1 @ T_2 @ T_3 @ T_4 @ T_5 @ T_6 @ T_7 @ T_8 @ T_9 @ T_10
print(T_test)
# model = pinocchio.buildSampleModelManipulator()
# data = model.createData()

JOINT_ID = 10
oMdes = pinocchio.SE3(np.eye(3), np.array([1., 0., 1.]))

q = np.zeros(10)
eps = 1e-4
IT_MAX = 1000
DT = 1e-1
damp = 1e-12

i = 0
while True:
    oMi_l, oMi_r = transform_mobilebasetoarmend(q[0:3], q[3:10], q[3:10])
    oMi = pinocchio.SE3(oMi_l[0:3, 0:3], oMi_l[0:3, 3])
    dMi = oMdes.actInv(oMi)
    err = pinocchio.log(dMi).vector
    if norm(err) < eps:
        success = True
        break
    if i >= IT_MAX:
        success = False
        break
    J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)
    v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
    q = pinocchio.integrate(model, q, v * DT)
    if not i % 10:
        print('%d: error = %s' % (i, err.T))
    i += 1

if success:
    print("Convergence achieved!")
else:
    print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

print('\nresult: %s' % q.flatten().tolist())
print('\nfinal error: %s' % err.T)
