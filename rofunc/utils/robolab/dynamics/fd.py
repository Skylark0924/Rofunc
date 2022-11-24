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
