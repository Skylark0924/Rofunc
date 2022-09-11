'''
    LQT computed in a recursive way (via-point example)

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Adie Niederberger <aniederberger@idiap.ch>,
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
from matplotlib.cm import get_cmap

# General parameters
# ===============================

param = {
	"nbData":100, # Number of datapoints
    "nbPoints": 2,  # Number of viapoints
    "nbDeriv" : 2, # Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
    "nbVarPos": 2,  # Dimension of position data (here: x1,x2)
    "dt" : 1E-2,  # Time step duration
    "rfactor": 1E-6,  # control cost in LQR
    }
param["nbVar"] = param["nbVarPos"] * param["nbDeriv"] # Dimension of state vector
param["nbVarX"] = param["nbVar"] + 1 # Augmented state space

# Task & System parameters
# ===============================
# Control cost matrix
R = np.eye(param["nbVarPos"]) * param["rfactor"]

# Dynamical System settings (discrete version)
A1d = np.zeros((param["nbDeriv"], param["nbDeriv"]))
for i in range(param["nbDeriv"]):
    A1d += np.diag(np.ones((param["nbDeriv"] - i,)), i) * param["dt"] ** i / np.math.factorial(i)  # Discrete 1D


B1d = np.zeros((param["nbDeriv"], 1))
for i in range(0, param["nbDeriv"]):
    B1d[param["nbDeriv"] - i - 1, :] = param["dt"] ** (i + 1) * 1 / np.math.factorial(i + 1)  # Discrete 1D

A0 = np.kron(A1d, np.eye(param["nbVarPos"]))  # Discrete nD
B0 = np.kron(B1d, np.eye(param["nbVarPos"]))  # Discrete nD
A = np.eye(A0.shape[0] + 1)  # Augmented A
A[:A0.shape[0], :A0.shape[1]] = A0
B = np.vstack((B0, np.zeros((1, param["nbVarPos"]))))  # Augmented B

# Sparse reference with a set of via-points
tl = np.linspace(0, param["nbData"] - 1, param["nbPoints"] + 1)
tl = np.rint(tl[1:])
Mu = np.array([[2,3],[5,1],[0,0],[0,0]])

# Definition of augmented precision matrix Qa based on covariance matrix Sigma0
# Sigma0 = np.diag(np.hstack((np.ones((param["nbVarPos"],)) * 1E-5, np.ones((param["nbVar"] - param["nbVarPos"],)) * 1E3)))
# Sigma = np.zeros((param["nbVar"] + 1, param["nbVar"] + 1, param["nbPoints"]))
# for i in range(param["nbPoints"]):
#     Sigma[:, :, i] = np.r_[np.c_[Sigma0 + Mu[:, i, np.newaxis] @ Mu[np.newaxis, :, i], Mu[:, i]],
#                           [np.append(Mu[:, i], [1])]] #Embedding of Mu in Sigma

# Q = np.zeros((param["nbVarX"], param["nbVarX"], param["nbData"]))
# for i in range(param["nbPoints"]):
#     Q[:, :, int(tl[i])] = np.linalg.inv(Sigma[:, :, i])

# Definition of augmented precision matrix Qa based on standard precision matrix Q0
Q0 = np.diag(
    np.hstack([
        np.ones(param["nbVarPos"]) , np.zeros(param["nbVar"]-param["nbVarPos"])
    ])
)  

Q0_augmented = np.identity(param["nbVar"]+1)
Q0_augmented[:param["nbVar"],:param["nbVar"]] = Q0

Q = np.zeros( [param["nbVar"]+1,param["nbVar"]+1,param["nbData"]] )
for i in range(param["nbPoints"]):
    Q[:, :, int(tl[i])] =  np.vstack([
        np.hstack([
            np.identity(param["nbVar"]),
            np.zeros([param["nbVar"],1])
        ]),
        np.hstack([
            -Mu[:,i].T,
            1
        ])
    ]) @ Q0_augmented @ np.vstack([
        np.hstack([
             np.identity(param["nbVar"]),
             -Mu[:,i].reshape([-1,1])
        ]),
        np.hstack([
            np.zeros(param["nbVar"]),
            1
        ])
    ])

# LQR with recursive computation and augmented state space
# ============================================================

state_noise = np.hstack(( -1 , -.1 , np.zeros(param["nbVar"]+1-param["nbVarPos"]) ))

P = np.zeros((param["nbVarX"], param["nbVarX"], param["nbData"]))
P[:, :, -1] = Q[:, :, -1]

r = np.zeros((param["nbVar"] + 1, 2, param["nbData"]))
for t in range(param["nbData"] - 2, -1, -1):
    P[:, :, t] = Q[:, :, t] - A.T @ (
                P[:, :, t + 1] @ np.dot(B, np.linalg.pinv(B.T @ P[:, :, t + 1] @ B + R))
                @ B.T @ P[:, :, t + 1] - P[:,:,t + 1]) @ A

# Reproduction with only feedback (FB) on augmented state
for n in range(2):

    x = np.hstack([ np.zeros(param["nbVar"]) , 1 ])
    for t in range(param["nbData"]):
        Z_bar = B.T @ P[:, :, t] @ B + R
        K = np.linalg.inv(Z_bar.T @ Z_bar) @ Z_bar.T @ B.T @ P[:, :, t] @ A  # Feedback gain
        u = -K @ x  # Acceleration command with FB on augmented state (resulting in feedback and feedforward terms)
        x = A @ x + B @ u  # Update of state vector

        if t==25 and n==1:
            x += state_noise

        r[:, n, t] = x # Log data


plt.figure()

cmap = get_cmap("Dark2")
cm = cmap.colors 

for n in range(2):
    plt.plot(r[0, n, :], r[1, n, :], c=cm[n], label="Trajectory {}".format(n+1))
    plt.scatter(r[0, n, 0], r[1, n, 0], marker='o',c=cm[n])

plt.scatter(Mu[0, :], Mu[1, :], s=20 * 1.5 ** 2, marker='o', color="red", label="Via-points")
plt.legend()
plt.show()