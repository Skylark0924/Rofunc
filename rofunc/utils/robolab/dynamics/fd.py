from __future__ import print_function

import numpy as np
import pinocchio


def fd(model, data, *args):
    model.lowerPositionLimit = -np.ones((model.nq, 1))
    model.upperPositionLimit = np.ones((model.nq, 1))

    if len(args) == 4 and type(args[3]) is bool:
        q = args[0]
        v = args[1]
        tau = args[2]
        external_force = args[3]
        if external_force:
            return pinocchio.aba(model, data, q, v, tau, external_force)
        else:
            return pinocchio.aba(model, data, q, v, tau)

    return pinocchio.aba(model, data, *args)


if __name__ == '__main__':
    model = pinocchio.buildModelFromUrdf(
        "/home/ubuntu/Rofunc/rofunc/simulator/assets/urdf/curi/urdf/curi_pinocchio_test.urdf")
    print('model name: ' + model.name)
    data = model.createData()

    q = pinocchio.randomConfiguration(model)
    q = pinocchio.normalize(model, q)
    v = np.matrix(np.random.rand(model.nv, 1))
    tau = np.matripinocchio.abax(np.random.rand(model.nv, 1))

    ddq = fd(model, data, q, v, tau)
    print(ddq)

    # pinocchio.computeABADerivatives(model, data, q, v, tau)
    #
    # ddq_dq = data.ddq_dq  # Derivatives of the FD w.r.t. the joint config vector
    # ddq_dv = data.ddq_dv  # Derivatives of the FD w.r.t. the joint velocity vector
    # ddq_dtau = data.Minv  # Derivatives of the FD w.r.t. the joint acceleration vector
