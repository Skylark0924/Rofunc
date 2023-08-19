"""
    iLQR for a 3D viapoints task (batch formulation)

    Refers to https://gitlab.idiap.ch/rli/robotics-codes-from-scratch by Dr. Sylvain Calinon
"""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from rofunc.config.utils import get_config
from omegaconf import DictConfig
from rofunc.utils.robolab.kinematics.fk import get_fk_from_model
from rofunc.planning_control.lqr.gluon_config import robot_config

kin = robot_config()


class iLQR_3D:
    def __init__(self, cgf):
        self.cfg = cgf

    def fkin(self, x):
        """
        Forward kinematics for end-effector (in robot coordinate system)

        """
        f = []
        for i in range(len(x)):
            forward = kin.fkine(x[i])

            f.append(forward)

        f = np.asarray(f)

        return f

    def error(self, f, f0):
        """
        Input Parameters:
        f = end_effector transformation matrix
        f0 = desired end_effector poses

        Output Parameters:
        error = position and orientation error
        """

        e_list = []
        for i in range(len(f0)):
            er = kin.error(f[i], f0[i])

            e_list.append(er)

        return e_list

    # def fkin0(self, x):
    #     """
    #     Forward kinematics for all individual links (in robot coordinate system)
    #     """
    #
    #     f = kin.fkinall(x)
    #
    #
    #     # f_all = np.asarray(f_all)
    #
    #     return f

    def Jacobian(self, x):
        """
        Jacobian with analytical computation (for single time step)

        Input Parameters:
        x = joint values

        Output Parameters:
        Jacobian for single time step of joint values x
        """
        J = kin.jacob(x)

        return J

    def f_reach(self, robot_state, Mu, Rot, specific_robot=None):
        """
        Error and Jacobian for a via-points reaching task (in object coordinate system)

        Input Parameters:
            robot_state: joint state
            Mu: via-points
            Rot: object orientation matrices

        Returns:
        f = residual vectors / error
        J = Jacobian of f

        """

        if specific_robot is not None:
            ee_pose = specific_robot.fkin(robot_state)
        else:
            ee_pose = self.fkin(robot_state)

        f = self.error(ee_pose, Mu)
        J = np.zeros([self.cfg.nbPoints * self.cfg.nbVarF, self.cfg.nbPoints * self.cfg.nbVarX])
        for t in range(self.cfg.nbPoints):
            f = np.asarray(f)
            f[t, :3] = Rot[t].T @ f[t, :3]  # Object-oriented forward kinematics

            Jtmp = self.Jacobian(robot_state[t])
            Jtmp[:3] = Rot[t].T @ Jtmp[:3]  # Object centered Jacobian

            # if self.cfg.useBoundingBox:
            #     for i in range(3):
            #         if abs(f[t, i]) < self.cfg.sz[i]:
            #             f[t, i] = 0
            #             Jtmp[i] = 0
            #         else:
            #             f[t, i] -= np.sign(f[t, i]) * self.cfg.sz[i]

            J[t * self.cfg.nbVarF:(t + 1) * self.cfg.nbVarF, t * self.cfg.nbVarX:(t + 1) * self.cfg.nbVarX] = Jtmp
        return f, J

    def get_matrices(self):
        # Precision matrix
        Q = np.identity(self.cfg.nbVarF * self.cfg.nbPoints)

        # Control weight matrix
        R = np.identity((self.cfg.nbData - 1) * self.cfg.nbVarU) * self.cfg.rfactor

        # Time occurrence of via-points
        tl = np.linspace(0, self.cfg.nbData, self.cfg.nbPoints + 1)
        tl = np.rint(tl[1:]).astype(np.int64) - 1
        idx = np.array([i + np.arange(0, self.cfg.nbVarX, 1) for i in (tl * self.cfg.nbVarX)])
        return Q, R, idx, tl

    def set_dynamical_system(self):
        # Transfer matrices (for linear system as single integrator)
        Su0 = np.vstack([np.zeros([self.cfg.nbVarX, self.cfg.nbVarX * (self.cfg.nbData - 1)]),
                         np.tril(np.kron(np.ones([self.cfg.nbData - 1, self.cfg.nbData - 1]),
                                         np.eye(self.cfg.nbVarX) * self.cfg.dt))])
        Sx0 = np.kron(np.ones(self.cfg.nbData), np.identity(self.cfg.nbVarX)).T
        return Su0, Sx0

    def get_u_x(self, Mu: np.ndarray, Rot: np.ndarray, u: np.ndarray, x0: np.ndarray, Q: np.ndarray,
                R: np.ndarray, Su0: np.ndarray, Sx0: np.ndarray, idx: np.ndarray, tl: np.ndarray):
        Su = Su0[idx.flatten()]  # We remove the lines that are out of interest

        for i in range(self.cfg.nbIter):
            x = Su0 @ u + Sx0 @ x0  # System evolution
            x = x.reshape([self.cfg.nbData, self.cfg.nbVarX])
            f, J = self.f_reach(x[tl], Mu, Rot)  # Residuals and Jacobians
            du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + R) @ (
                    -Su.T @ J.T @ Q @ f.flatten() - u * self.cfg.rfactor)  # Gauss-Newton update
            # Estimate step size with backtracking line search method
            alpha = 1  # 2
            cost0 = f.flatten() @ Q @ f.flatten() + np.linalg.norm(u) ** 2 * self.cfg.rfactor  # Cost
            while True:
                utmp = u + du * alpha
                xtmp = Su0 @ utmp + Sx0 @ x0  # System evolution
                # print(xtmp.shape)
                # print(xtmp)
                xtmp = xtmp.reshape([self.cfg.nbData, self.cfg.nbVarX])
                ftmp, _ = self.f_reach(xtmp[tl], Mu, Rot)  # Residuals
                cost = ftmp.flatten() @ Q @ ftmp.flatten() + np.linalg.norm(utmp) ** 2 * self.cfg.rfactor  # Cost
                if cost < cost0 or alpha < 1e-3:
                    u = utmp
                    print("Iteration {}, cost: {}".format(i, cost))
                    print(np.linalg.norm(du * alpha))
                    break
                alpha /= 2
            if np.linalg.norm(du * alpha) < 1E-2:
                break  # Stop iLQR iterations when solution is reached
        return u, x

    def solve(self, Mu, Rot, u0, x0, for_test=False):
        Q, R, idx, tl = self.get_matrices()
        Su0, Sx0 = self.set_dynamical_system()
        u, x = self.get_u_x(Mu, Rot, u0, x0, Q, R, Su0, Sx0, idx, tl)
        # self.vis(Mu, Rot, x, tl, for_test=for_test)

        return u, x
