import pinocchio
import numpy as np


model = pinocchio.buildModelFromUrdf(
        "/home/ubuntu/Rofunc/rofunc/simulator/assets/urdf/curi/urdf/curi_pinocchio_test.urdf")
print('model name: ' + model.name)
data = model.createData()

model.lowerPositionLimit = -np.ones((model.nq, 1))
model.upperPositionLimit = np.ones((model.nq, 1))

q = pinocchio.randomConfiguration(model)  # joint configuration
v = np.random.rand(model.nv, 1)  # joint velocity
a = np.random.rand(model.nv, 1)  # joint acceleration
tau = pinocchio.rnea(model, data, q, v, a)
print(tau)

# pinocchio.computeRNEADerivatives(model, data, q, v, a)
#
# dtau_dq = data.dtau_dq  # Derivatives of the ID w.r.t. the joint config vector
# dtau_dv = data.dtau_dv  # Derivatives of the ID w.r.t. the joint velocity vector
# dtau_da = data.M  # Derivatives of the ID w.r.t. the joint acceleration vector
