from __future__ import print_function
from fk_new_config import transform_mobilebasetoarmend
import numpy as np
import pinocchio
from numpy.linalg import norm, solve

model = pinocchio.buildSampleModelManipulator()
data = model.createData()

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
