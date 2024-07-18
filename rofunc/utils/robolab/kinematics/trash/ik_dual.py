from __future__ import print_function

import numpy as np
from numpy.linalg import norm, solve

eps = 1e-4
eps_r = 1e-4
IT_MAX = 10000
DT = 1e-1
damp = 1e-12


def ik_dual(model, POSE_L, POSE_R, JOINT_ID_L, JOINT_ID_R):
    import pinocchio

    data = model.createData()

    oMdes_l = pinocchio.SE3(np.eye(3), np.array(POSE_L))
    oMdes_r = pinocchio.SE3(np.eye(3), np.array(POSE_R))
    q = pinocchio.neutral(model)

    i = 0
    while True:
        pinocchio.forwardKinematics(model, data, q)
        dMi_l = oMdes_l.actInv(data.oMi[JOINT_ID_L])
        err_l = pinocchio.log(dMi_l).vector
        dMi_r = oMdes_r.actInv(data.oMi[JOINT_ID_R])
        err_r = pinocchio.log(dMi_r).vector
        if norm(err_l) < eps and norm(err_r) < eps_r:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        J_l = pinocchio.computeJointJacobian(model, data, q, JOINT_ID_L)
        v_l = - J_l.T.dot(solve(J_l.dot(J_l.T) + damp * np.eye(6), err_l))
        J_r = pinocchio.computeJointJacobian(model, data, q, JOINT_ID_R)
        v_r = - J_r.T.dot(solve(J_r.dot(J_r.T) + damp * np.eye(6), err_r))
        q = pinocchio.integrate(model, q, 0.5 * v_l * DT + 0.5 * v_r * DT)
        if not i % 10:
            print('%d: error = %s' % (i, err_l.T))
            print('%d: error = %s' % (i, err_r.T))
        i += 1

    if success:
        print("Convergence achieved!")
    else:
        print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

    q_ik_dual = np.append(0, np.delete(q, [1, 3, 5, 7]))
    i = 0
    for name, value in zip(model.names, q_ik_dual):
        print(("{: .0f} {:<24} : {: .4f}"
               .format(i, name, value)))
        i += 1
    print('\nresult: %s' % q_ik_dual.flatten().tolist())
    print('\nfinal error: %s, %s' % (err_l.T, err_r.T))
    return q_ik_dual


# if __name__ == '__main__':
#     model = pinocchio.buildModelFromUrdf(
#         "/home/ubuntu/Rofunc/rofunc/simulator/assets/urdf/curi/urdf/curi_pinocchio_test.urdf")
#     print('model name: ' + model.name)
#     POSE_L = [1, 0.5, 0.5]
#     POSE_R = [1, -0.5, 0.5]
#     q_rearrange = ik_dual(model, POSE_L, POSE_R, JOINT_ID_L=18, JOINT_ID_R=27)
#     a = q_rearrange.take(
#         [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 21, 11, 13, 22, 23, 14, 15, 24, 16, 25, 26, 17, 18, 27, 19, 28])
#     print('\nresult: %s' % a.flatten().tolist())
