'''
    batch iLQR applied on a planar manipulator for a tracking problem involving the center of mass (CoM) and the end-effector

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Teng Xue <teng.xue@idiap.ch>,
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
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Helper functions
# ===============================
# Forward kinematics (in robot coordinate system)
def fkin(x):
    global param
    x = x.T
    A = np.tril(np.ones([param.nbVarX, param.nbVarX]))
    f = np.vstack((param.l @ np.cos(A @ x),
                   param.l @ np.sin(A @ x)))
    return f

# Forward Kinematics for all joints
def fkin0(x):
    T = np.tril(np.ones([param.nbVarX, param.nbVarX]))
    T2 = np.tril(np.matlib.repmat(param.l, len(x), 1))
    f = np.vstack((
        T2 @ np.cos(T @ x),
        T2 @ np.sin(T @ x)
    )).T
    f = np.vstack((
        np.zeros(2),
        f
    ))
    return f

# Jacobian with analytical computation (for single time step)
def jkin(x):
    global param
    T = np.tril(np.ones((len(x), len(x))))
    J = np.vstack((
        -np.sin(T @ x).T @ np.diag(param.l) @ T,
        np.cos(T @ x).T @ np.diag(param.l) @ T
    ))
    return J

# Cost and gradient for end-effector
def f_reach(x):
    global param

    f = fkin(x).T - param.Mu
    J = np.zeros((len(x) * param.nbVarF, len(x) * param.nbVarX))

    for t in range(x.shape[0]):
        Jtmp = jkin(x[t])
        J[t * param.nbVarF:(t + 1) * param.nbVarF, t * param.nbVarX:(t + 1) * param.nbVarX] = Jtmp
    return f.T, J

# Forward kinematics for center of mass (in robot coordinate system, with mass located at the joints)
def fkin_CoM(x):
    global param
    x = x.T
    A = np.tril(np.ones([param.nbVarX, param.nbVarX]))
    f = np.vstack((param.l @ A @ np.cos(A @ x),
                   param.l @ A @ np.sin(A @ x))) / param.nbVarX
    return f

# Cost and gradient for center of mass
def f_reach_CoM(x):
    global param
    f = fkin_CoM(x).T - param.MuCoM
    J = np.zeros((len(x) * param.nbVarF, len(x) * param.nbVarX))
    A = np.tril(np.ones([param.nbVarX, param.nbVarX]))
    for t in range(x.shape[0]):
        Jtmp = np.vstack((-np.sin(A @ x[t] ).T @ A @ np.diag(param.l @ A) ,
                    np.cos(A @ x[t] ).T @ A @ np.diag(param.l @ A)))/param.nbVarX
        if param.useBoundingBox:
            for i in range(1):
                if abs(f[t, i]) < param.szCoM:
                    f[t, i] = 0
                    Jtmp[i] = 0
                else:
                    f[t, i] -= np.sign(f[t, i]) * param.szCoM
        J[t * param.nbVarF:(t + 1) * param.nbVarF, t * param.nbVarX:(t + 1) * param.nbVarX] = Jtmp
    f = f.flatten().T
    return f, J


# Parameters
class Param:
    def __init__(self):
        self.dt = 1e-1 # Time step length
        self.nbData = 10 # Number of datapoints
        self.nbIter = 50 # Maximum number of iterations for iLQR
        self.nbPoints = 1 # Number of viapoints
        self.nbVarX = 5 # State space dimension (x1,x2,x3)
        self.nbVarU = 5 # Control space dimension (dx1,dx2,dx3)
        self.nbVarF = 2 # Objective function dimension (f1,f2,f3, with f3 as orientation)
        self.l = [2, 2, 2, 2, 2] # Robot links lengths
        self.szCoM = .6
        self.useBoundingBox = True
        self.r = 1e-5 # Control weight term
        self.Mu = np.asarray([3.5, 4]) # Target
        self.MuCoM = np.asarray([.4, 0])


# Main program
# ===============================
param = Param()

# Task parameters
# ===============================
# Regularization matrix
R = np.identity((param.nbData - 1) * param.nbVarU) * param.r

# Precision matrix
Q = np.identity(param.nbVarF * param.nbPoints)

# Precision matrix for CoM (by considering only horizonal CoM location)
Qc = np.kron(np.identity(param.nbData), np.diag([1E0, 0]))

# System parameters
# ===============================
# Time occurence of viapoints
tl = np.linspace(0, param.nbData, param.nbPoints + 1)
tl = np.rint(tl[1:]).astype(np.int64) - 1
idx = np.array([ i + np.arange(0,param.nbVarX,1) for i in (tl* param.nbVarX)])

u = np.zeros(param.nbVarU * (param.nbData - 1))  # Initial control command
a = .7
x0 = np.array([np.pi / 2 - a, 2 * a, - a, np.pi - np.pi / 4, 3 * np.pi / 4])  # Initial state (in joint space)

# Transfer matrices (for linear system as single integrator)
Su0 = np.vstack([np.zeros((param.nbVarX, param.nbVarX * (param.nbData - 1))),
                 np.tril(np.kron(np.ones((param.nbData - 1, param.nbData - 1)),
                                 np.eye(param.nbVarX) * param.dt))])
Sx0 = np.kron(np.ones(param.nbData), np.identity(param.nbVarX)).T
Su = Su0[idx.flatten()]  # We remove the lines that are out of interest

# Solving iLQR
# ===============================
for i in range(param.nbIter):
    x = Su0 @ u + Sx0 @ x0
    x = x.reshape((param.nbData, param.nbVarX))

    f, J = f_reach(x[tl])
    fc, Jc = f_reach_CoM (x)
    du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + Su0.T @ Jc.T @ Qc @ Jc @ Su0 + R) @ (-Su.T @ J.T @ Q @ f.flatten() - Su0.T @ Jc.T @ Qc @ fc.flatten() - u * param.r)

    # Perform line search
    alpha = 1
    cost0 = f.flatten() @ Q @ f.flatten() + fc.flatten() @ Qc @ fc.flatten() + np.linalg.norm(u) * param.r

    while True:
        utmp = u + du * alpha
        xtmp = Su0 @ utmp + Sx0 @ x0
        xtmp = xtmp.reshape((param.nbData, param.nbVarX))
        ftmp, _ = f_reach(xtmp[tl])
        fctmp, _ = f_reach_CoM(xtmp)
        cost = ftmp.flatten() @ Q @ ftmp.flatten() + fctmp. T @ Qc @ fctmp + np.linalg.norm(utmp) * param.r

        if cost < cost0 or alpha < 1e-3:
            u = utmp
            print("Iteration {}, cost: {}, alpha: {}".format(i, cost, alpha))
            break

        alpha /= 2

# Ploting
# ===============================
plt.figure()
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

# plot ground
plt.plot([-1, 3], [0, 0], linestyle='-', c=[.2, .2, .2], linewidth=2)

# Get points of interest
f00 = fkin0(x[0])
fT0 = fkin0(x[-1])
fc = fkin_CoM(x)

plt.plot(f00[:, 0], f00[:, 1], c=[.8, .8, .8], linewidth=4, linestyle='-')
plt.plot(fT0[:, 0], fT0[:, 1], c=[.4, .4, .4], linewidth=4, linestyle='-')

#plot CoM
plt.plot(fc[0, 0], fc[1, 0], c=[.5, .5, .5], marker="o", markeredgewidth=4, markersize=8, markerfacecolor='white')
plt.plot(fc[0, -1], fc[1, -1], c=[.2, .2, .2], marker="o", markeredgewidth=4, markersize=8, markerfacecolor='white')

#plot end-effector target
plt.plot(param.Mu[0], param.Mu[1], marker="o", markersize=8, c="r")

# Plot bounding box or via-points
ax = plt.gca()
for i in range(param.nbPoints):
    if param.useBoundingBox:
        rect_origin = param.MuCoM + np.array([0, 3.5]) - np.array([param.szCoM, 3.5])
        rect = patches.Rectangle(rect_origin, param.szCoM * 2, 3.5 * 2,
                                 facecolor=[.8, 0, 0], alpha=0.1, edgecolor=None)
        ax.add_patch(rect)
plt.show()
