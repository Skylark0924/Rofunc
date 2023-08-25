"""
CURI inverse kinematics
========================

Inverse kinematics of the CURI robot.
"""
import os

import numpy as np
import pinocchio

import rofunc as rf
from rofunc.utils.oslab.path import get_rofunc_path

model = pinocchio.buildModelFromUrdf(
    os.path.join(get_rofunc_path(), "simulator/assets/urdf/curi/urdf/curi_pinocchio_test.urdf"))
print('model name: ' + model.name)

POSE = [1, 0, 1]
ORI = np.array([-1, 0, 0, 0, -1, 0, 0, 0, 1]).reshape(3, 3)
q_ik = rf.robolab.ik(model, POSE, ORI, JOINT_ID=18)

POSE_L = [1, 0.5, 0.5]
POSE_R = [1, -0.5, 0.5]
q_ik_dual = rf.robolab.ik_dual(model, POSE_L, POSE_R, JOINT_ID_L=18, JOINT_ID_R=27)

# a = q_ik_dual.take(
#     [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 21, 11, 13, 22, 23, 14, 15, 24, 16, 25, 26, 17, 18, 27, 19, 28])
# print('\nresult: %s' % a.flatten().tolist())
