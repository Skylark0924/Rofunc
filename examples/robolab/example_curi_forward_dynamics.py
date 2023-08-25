"""
CURI forward dynamics
========================

Forward dynamics of the CURI robot.
"""
import os

import numpy as np
import pinocchio

import rofunc as rf
from rofunc.utils.oslab.path import get_rofunc_path

model = pinocchio.buildModelFromUrdf(
    os.path.join(get_rofunc_path(), "simulator/assets/urdf/curi/urdf/curi_pinocchio_test.urdf"))
print('model name: ' + model.name)
data = model.createData()

q = pinocchio.randomConfiguration(model)
q = pinocchio.normalize(model, q)
v = np.matrix(np.random.rand(model.nv, 1))
tau = np.matrix(np.random.rand(model.nv, 1))

ddq = rf.robolab.fd(model, data, q, v, tau)
print(ddq)

# pinocchio.computeABADerivatives(model, data, q, v, tau)
#
# ddq_dq = data.ddq_dq  # Derivatives of the FD w.r.t. the joint config vector
# ddq_dv = data.ddq_dv  # Derivatives of the FD w.r.t. the joint velocity vector
# ddq_dtau = data.Minv  # Derivatives of the FD w.r.t. the joint acceleration vector
