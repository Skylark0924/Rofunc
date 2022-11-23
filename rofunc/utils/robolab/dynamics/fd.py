from __future__ import print_function

import numpy as np
import pinocchio


def fd(model, data, *args):
    if len(args) == 7 and type(args[6]) is bool:
        q = args[0]
        v = args[1]
        tau = args[2]
        J = args[3]
        gamma = args[4]
        inv_damping = args[5]
        updateKinematics = args[6]
        if updateKinematics:
            return pinocchio.forwardDynamics(model, data, q, v, tau, J, gamma, inv_damping)
        else:
            return pinocchio.forwardDynamics(model, data, tau, J, gamma, inv_damping)

    return pinocchio.forwardDynamics(model, data, *args)


if __name__ == '__main__':
    model = pinocchio.buildModelFromUrdf(
        "/home/ubuntu/Rofunc/rofunc/simulator/assets/urdf/curi/urdf/curi_pinocchio_test.urdf")
    print('model name: ' + model.name)
    data = model.createData()

    model.lowerPositionLimit = -np.ones((model.nq, 1))
    model.upperPositionLimit = np.ones((model.nq, 1))

    q = pinocchio.randomConfiguration(model)
    q = pinocchio.normalize(model, q)
    v = np.matrix(np.random.rand(model.nv, 1))
    tau = np.matrix(np.random.rand(model.nv, 1))

    a = pinocchio.aba(model, data, q, v, tau)
    print(a)

    pinocchio.computeABADerivatives(model, data, q, v, tau)

    ddq_dq = data.ddq_dq  # Derivatives of the FD w.r.t. the joint config vector
    ddq_dv = data.ddq_dv  # Derivatives of the FD w.r.t. the joint velocity vector
    ddq_dtau = data.Minv  # Derivatives of the FD w.r.t. the joint acceleration vector
    print(ddq_dtau)
