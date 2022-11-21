from __future__ import print_function

import numpy as np
from numpy.linalg import norm, solve
from sys import argv
from os.path import dirname, join, abspath

import pinocchio

# model = pinocchio.buildSampleModelManipulator()
# data = model.createData()

# Load the urdf model
# model = pinocchio.buildModelFromUrdf("/home/lee/Rofunc/rofunc/utils/robolab/franka_description/robots/franka_panda.urdf")
model = pinocchio.buildModelFromUrdf("/home/lee/Rofunc/rofunc/simulator/assets/urdf/curi/urdf/curi_pinocchio_test.urdf")
print('model name: ' + model.name)

# Create data required by the algorithms
data = model.createData()

# Sample a random configuration
q = pinocchio.randomConfiguration(model)
print('q: %s' % q.T)

# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(model, data, q)

# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}"
           .format(name, *oMi.translation.T.flat)))

JOINT_ID = 9
oMdes = pinocchio.SE3(np.eye(3), np.array([-0.2, 0.2, 0.1]))

q = pinocchio.neutral(model)
eps = 1e-4
IT_MAX = 1000
DT = 1e-1
damp = 1e-12

i = 0
while True:
    pinocchio.forwardKinematics(model, data, q)
    dMi = oMdes.actInv(data.oMi[JOINT_ID])
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
