from __future__ import print_function

import numpy as np
import pinocchio
from numpy.linalg import norm, solve

eps = 1e-4
IT_MAX = 10000
DT = 1e-1
damp = 1e-12

limit = [[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698], [-2.8973, 2.8973],
         [-0.0175, 3.7525], [-2.8973, 2.8973]]


def ik(model, position, orientation, JOINT_ID):
    data = model.createData()

    oMdes = pinocchio.SE3(orientation, np.array(position))
    q = pinocchio.neutral(model)
    i = 0
    while True:
        pinocchio.forwardKinematics(model, data, q)
        iMd = data.oMi[JOINT_ID].actInv(oMdes)
        err = pinocchio.log(iMd).vector
        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)
        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
        v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pinocchio.integrate(model, q, v * DT)
        for j in range(len(q)):
            if q[j] < limit[j][0]:
                q[j] = limit[j][0]
            elif q[j] > limit[j][1]:
                q[j] = limit[j][1]

        if not i % 10:
            print('%d: error = %s' % (i, err.T))
        i += 1

    if success:
        print("Convergence achieved!")
    else:
        print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

    # q_ik = np.append(0, q, )
    # i = 0
    # for name, value in zip(model.names, q_ik):
    #     print(("{: .0f} {:<24} : {: .4f}"
    #            .format(i, name, value)))
    #     i += 1
    print('\nresult: %s' % q.flatten().tolist())
    print('\nfinal error: %s' % err.T)
    return q


if __name__ == '__main__':
    model = pinocchio.buildModelFromUrdf(
        "/home/ubuntu/Github/Rofunc/rofunc/simulator/assets/urdf/franka_description/robots/franka_panda_no_gripper.urdf")
    print('model name: ' + model.name)
    position = [1, 0, 1]
    orientation = np.eye(3)
    JOINT_ID = 7
    ik(model, position, orientation, JOINT_ID)
