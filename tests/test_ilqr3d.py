import rofunc as rf
import numpy as np
from rofunc.config.utils import get_config
from scipy.spatial.transform import Rotation as R
from rofunc.planning_control.lqr.ilqr_3d import iLQR_3D
import pandas as pd



def test_7d_uni_ilqr_3D():
    cfg = get_config('./planning', 'ilqr3d')
    # via-points
    # Mu = np.array([[2, 1, -np.pi / 6], [3, 2, -np.pi / 3]])  # Via-points
    # Mu = np.array([[7.5685673e-02, -2.2696000e-05, 5.3397900e-01, 0, 0, 0, 1],
    Mu = np.array([[0.10021176, -0.01036865, 0.49858453, 0.61920622, 0.19450308, 0.21959119, 0.72837622],
                   [0.08177911, -0.06516777, 0.44698613, 0.88028369, 0.03904804, 0.02095377, 0.4723736],
                   [0.07767701, -0.04641878, 0.4275838, 0.79016704, 0.01637976, 0.01269766, 0.61254103],
                   [0.06642697, 0.28006863, 0.39004221, 0.34475831, 0.01169578, 0.01790368, 0.93844785]])

    Rot = np.zeros([cfg.nbPoints, 3, 3])  # Object orientation matrices
    # Object rotation matrices
    for t in range(cfg.nbPoints):
        orn_t = Mu[t, 3:]

        quat = R.from_quat(orn_t)

        Rot[t] = quat.as_matrix()

    u0 = np.zeros(cfg.nbVarU * (cfg.nbData - 1))  # Initial control command
    x0 = np.array([0, 0, 0, 0, 0, 0])  # Initial state
    controller = iLQR_3D(cfg)
    u, x = controller.solve(Mu, Rot, u0, x0, for_test=True)

    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]
    x6 = x[:, 5]

    command = []
    for i in range(int(len(u)/6)):

        com = np.array(u[i*6:i*6+6])
        command.append(com)

        i = i+1

    command = np.asarray(command)

    print(x)

    u1 = command[:, 0]
    u2 = command[:, 1]
    u3 = command[:, 2]
    u4 = command[:, 3]
    u5 = command[:, 4]
    u6 = command[:, 5]

    df = pd.DataFrame(list(zip(x1, x2, x3, x4, x5, x6, u1, u2, u3, u4, u5, u6)), columns=['J_dot_1', 'J_dot_2', 'J_dot_3', 'J_dot_4', 'J_dot_5', 'J_dot_6', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6'])
    df.to_excel('/home/ubuntu/Github/clover-innfos-python/trajectories/test_traj/test_ilqr3d_02s.xlsx', columns=['J_dot_1', 'J_dot_2', 'J_dot_3', 'J_dot_4', 'J_dot_5', 'J_dot_6', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6'])

if __name__ == '__main__':
    test_7d_uni_ilqr_3D()