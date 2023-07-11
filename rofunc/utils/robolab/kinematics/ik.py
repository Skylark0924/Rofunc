from __future__ import print_function

import numpy as np
import pinocchio
from numpy.linalg import norm, solve

eps = 1e-6
IT_MAX = 10000
DT = 1e-1
damp = 1e-12


def ik(model, POSE, ORI, JOINT_ID):
    data = model.createData()

    oMdes = pinocchio.SE3(ORI, np.array(POSE))
    q = pinocchio.neutral(model)
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

    q_ik = np.append(0, np.delete(q, [1, 3, 5, 7]))
    i = 0
    for name, value in zip(model.names, q_ik):
        print(("{: .0f} {:<24} : {: .4f}"
               .format(i, name, value)))
        i += 1
    print('\nresult: %s' % q_ik.flatten().tolist())
    print('\nfinal error: %s' % err.T)
    return q_ik


if __name__ == '__main__':
    model = pinocchio.buildModelFromUrdf(
        "/home/lee/Rofunc/rofunc/simulator/assets/urdf/curi/urdf/curi_pinocchio_test.urdf")
    print('model name: ' + model.name)
    POSE = [1, 0, 1]
    ORI = np.array([-1, 0, 0, 0, -1, 0, 0, 0, 1]).reshape(3, 3)
    JOINT_ID = 18
    q_rearrange = ik(model, POSE, ORI, JOINT_ID)
    a = q_rearrange.take(
        [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 21, 11, 13, 22, 23, 14, 15, 24, 16, 25, 26, 17, 18, 27, 19, 28])
    print('\nresult: %s' % a.flatten().tolist())
