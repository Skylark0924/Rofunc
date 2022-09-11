'''
    iLQR applied to a planar bimanual robot for a tracking problem involving
    the center of mass (CoM) and the end-effector (batch formulation)

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Adi Niederberger <aniederberger@idiap.ch> and
    Sylvain Calinon <https://calinon.ch>

    This file is part of RCFS.

    RCFS is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3 as
    published by the Free Software Foundation.

    RCFS is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with RCFS. If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Helper functions
# ===============================
def fkin(x):
    L = np.tril(np.ones(3))
    x = x[np.newaxis, :] if x.ndim == 1 else x
    f = np.vstack((param.l[:3] @ np.cos(L @ x[:, :3].T),
                   param.l[:3] @ np.sin(L @ x[:, :3].T),
                   np.array(param.l)[[0, 3, 4]] @ np.cos(L @ x[:, (0, 3, 4)].T),
                   np.array(param.l)[[0, 3, 4]] @ np.sin(L @ x[:, (0, 3, 4)].T)
                   ))
    return f


def fkin0(x):
    L = np.tril(np.ones(3))
    fl = np.vstack((L @ np.diag(param.l[:3]) @ np.cos(L @ x[:3].T),
                    L @ np.diag(param.l[:3]) @ np.sin(L @ x[:3].T)
                    ))

    fr = np.vstack((L @ np.diag(np.array(param.l)[[0, 3, 4]]) @ np.cos(L @ x[[0, 3, 4]].T),
                    L @ np.diag(np.array(param.l)[[0, 3, 4]]) @ np.sin(L @ x[[0, 3, 4]].T)
                    ))

    f = np.hstack((fl[:, ::-1], np.zeros((2, 1)), fr))
    return f


def f_reach(x):
    f = fkin(x) - param.Mu
    f = f.ravel()
    J = np.zeros((param.nbVarF * x.shape[0], x.shape[0] * param.nbVarX))
    for t in range(x.shape[0]):
        Jtmp = Jkin(x[t])
        J[t * param.nbVarF:(t + 1) * param.nbVarF, t * param.nbVarX:(t + 1) * param.nbVarX] = Jtmp
    return f, J


def Jkin(x):
    L = np.tril(np.ones(3))
    J = np.zeros((param.nbVarF, param.nbVarX))
    Ju = np.vstack((-np.sin(L @ x[:3]) @ np.diag(param.l[:3]) @ L,
                    np.cos(L @ x[:3]) @ np.diag(param.l[:3]) @ L
                    ))

    Jl = np.vstack((-np.sin(L @ x[[0, 3, 4]]) @
                    np.diag(np.array(param.l)[[0, 3, 4]]) @ L,
                    np.cos(L @ x[[0, 3, 4]]) @
                    np.diag(np.array(param.l)[[0, 3, 4]]) @ L
                    ))
    J[:Ju.shape[0], :Ju.shape[1]] = Ju
    J[2:, (0, 3, 4)] = Jl
    return J


def f_reach_CoM(x):
    f = fkin_CoM(x) - np.array([param.MuCoM]).T
    f = f.ravel(order="F")

    J = np.zeros((2 * x.shape[0], x.shape[0] * param.nbVarX))
    for t in range(x.shape[0]):
        Jtmp = Jkin_CoM(x[t])
        J[t * 2:(t + 1) * 2, t * param.nbVarX:(t + 1) * param.nbVarX] = Jtmp
    return f, J


def fkin_CoM(x):
    L = np.tril(np.ones(3))
    f = np.vstack((param.l[:3] @ L @ np.cos(L @ x[:, :3].T) +
                   np.array(param.l)[[0, 3, 4]] @ L @ np.cos(L @ x[:, (0, 3, 4)].T),
                   param.l[:3] @ L @ np.sin(L @ x[:, :3].T) +
                   np.array(param.l)[[0, 3, 4]] @ L @ np.sin(L @ x[:, (0, 3, 4)].T)
                   )) / 6
    return f


def Jkin_CoM(x):
    L = np.tril(np.ones(3))
    Jl = np.vstack((-np.sin(L @ x[:3]) @ L @ np.diag(param.l[:3] @ L),
                    np.cos(L @ x[:3]) @ L @ np.diag(param.l[:3] @ L)
                    )) / 6
    Jr = np.vstack((-np.sin(L @ x[[0, 3, 4]]) @ L @ np.diag(np.array(param.l)[[0, 3, 4]] @ L),
                    np.cos(L @ x[[0, 3, 4]]) @ L @ np.diag(np.array(param.l)[[0, 3, 4]] @ L)
                    )) / 6
    J = np.hstack(((Jl[:, 0] + Jr[:, 0])[:, np.newaxis], Jl[:, 1:], Jr[:, 1:]))
    return J


# Parameters
class Param:
    def __init__(self):
        self.dt = 1e0 # Time step length
        self.nbData = 30 # Number of datapoints
        self.nbIter = 100 # Maximum number of iterations for iLQR
        self.nbPoints = 1 # Number of viapoints
        self.nbVarX = 5 # State space dimension ([q1,q2,q3] for left arm, [q1,q4,q5] for right arm)
        self.nbVarU = self.nbVarX # Control space dimension (dq1,dq2,dq3,dq4,dq5)
        self.nbVarF = 4 # Task space dimension ([x1,x2] for left end-effector, [x3,x4] for right end-effector)
        self.l = [2]*self.nbVarX # Robot links lengths
        self.r = 1e-5 # Control weight term
        self.Mu = np.array([[-1, -1.5, 4, 2 ]]).T # Target point for end-effectors
        self.MuCoM = np.array([0, 1.4]) # Target point for center of mass

# Main program
# ===============================

param = Param()

# Control weight matrix (at trajectory level)
R = np.eye(param.nbVarU * (param.nbData-1)) * param.r
# Precision matrix for end-effectors tracking
Q = np.kron(np.eye(param.nbPoints), np.diag([1, 1, 0, 0]))
# Precision matrix for continuous CoM tracking
Qc = np.kron(np.eye(param.nbData), np.diag([1, 1]))

# Time occurence of viapoints
tl = np.linspace(0, param.nbData-1, param.nbPoints+1 )
tl = np.round(tl[1:]).astype(np.int32)
idx = (tl - 1)[:,np.newaxis] * param.nbVarX + np.arange(param.nbVarU)

# initial setup
u = np.zeros( param.nbVarU * (param.nbData-1) ) # Initial control command
x0 = np.array( [np.pi/2, np.pi/2, np.pi/3, -np.pi/2, -np.pi/3] )#Initial pose

# Solving Iterative LQR (iLQR)
# ===============================

# Transfer matrices (for linear system as single integrator)
Su0 = np.vstack([np.zeros((param.nbVarX, param.nbVarX*(param.nbData-1))),
      np.tril(np.kron(np.ones((param.nbData-1, param.nbData-1)), np.eye(param.nbVarX)*param.dt))])
Sx0 = np.kron( np.ones(param.nbData) , np.identity(param.nbVarX) ).T
Su = Su0[idx.flatten()] # We remove the lines that are out of interest

for i in range(param.nbIter):
    x = Su0 @ u + Sx0 @ x0
    x = x.reshape((param.nbData, param.nbVarX))
    f, J = f_reach(x[tl])  # Forward kinematics and Jacobian for end-effectors
    fc, Jc = f_reach_CoM(x)  # Forward kinematics and Jacobian for center of mass

    du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + Su0.T @ Jc.T @ Qc @ Jc @ Su0 + R) @ \
         (-Su.T @ J.T @ Q @ f - Su0.T @ Jc.T @ Qc @ fc - u * param.r)

    # Estimate step size with line search method
    alpha = 1
    cost0 = f.T @ Q @ f + fc.T @ Qc @ fc + np.linalg.norm(u) ** 2 * param.r

    while True:
        utmp = u + du * alpha
        xtmp = (Su0 @ utmp + Sx0 @ x0).reshape((param.nbData, param.nbVarX))
        ftmp, _ = f_reach(xtmp[tl])
        fctmp, _ = f_reach_CoM(xtmp)

        # for end-effectors and CoM
        cost = ftmp.T @ Q @ ftmp + fctmp.T @ Qc @ fctmp + np.linalg.norm(utmp) ** 2 * param.r # for end-effectors and CoM
        if cost < cost0 or alpha < 1e-3:
            print("Iteration {}, cost: {}, alpha: {}".format(i, cost, alpha))
            break # Stop iLQR when solution is reached

        alpha *= .5

    u = u + du * alpha
    if np.linalg.norm(du * alpha) < 1e-2:
        break

#  Plot state space
tl = np.array([0, tl.item()])

plt.figure(figsize=(15, 9))
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

# Plot bimanual robot
ftmp = fkin0(x[0])
plt.plot(ftmp[0], ftmp[1], c='black', linewidth=4, alpha=.2)

ftmp = fkin0(x[-1])
plt.plot(ftmp[0], ftmp[1], c='black', linewidth=4, alpha=.6)

# Plot CoM
fc = fkin_CoM(x)  # Forward kinematics for center of mass
plt.plot(fc[0, 0], fc[1, 0], c='black', marker='o', linewidth=0,
         markersize=np.sqrt(90), markerfacecolor='none', markeredgewidth=3, alpha=.4)  # Plot CoM
plt.plot(fc[0, -1], fc[1, -1], c='black', marker='o', linewidth=0,
         markersize=np.sqrt(90), markerfacecolor='none', markeredgewidth=3, alpha=.6)  # Plot CoM

# Plot end-effectors targets
for t in range(param.nbPoints):
    plt.plot(param.Mu[0, t], param.Mu[1, t], marker='o', c='red', markersize=14)

# Plot CoM target
plt.plot(param.MuCoM[0], param.MuCoM[1], c='red', marker='o', linewidth=0,
         markersize=np.sqrt(90), markerfacecolor='none', markeredgewidth=2, alpha=.8)

# Plot end-effectors paths
ftmp = fkin(x)
plt.plot(ftmp[0, :], ftmp[1, :], c="black", marker="o", markevery=[0] + tl.tolist())
plt.plot(ftmp[2, :], ftmp[3, :], c="black", marker="o", markevery=[0] + tl.tolist())
plt.show()


