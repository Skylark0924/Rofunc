'''
    Linear Quadratic tracker applied on a via point example

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Jeremy Maceiras <jeremy.maceiras@idiap.ch>,
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
from math import factorial
import matplotlib.pyplot as plt

# Parameters
# ===============================
param = {
    "nbData" : 100, # Number of data points
    "nbPoints" : 3, # Number of viapoints
    "nbVarPos" : 2, # Dimension of position data
    "nbDeriv" : 2,  # Number of static and dynamic features (2 -> [x,dx])
    "dt" : 1e-2, # Time step duration
    "rfactor" : 1e-8 # Control cost
}

nb_var = param["nbVarPos"] * param["nbDeriv"] # Dimension of state vector
R = np.identity( (param["nbData"]-1) * param["nbVarPos"] ) * param["rfactor"]  # Control cost matrix

#param["muQ"] = np.vstack((  # Sparse reference
#    np.zeros(((param["nbData"]-1) * nb_var , 1)),
#    np.vstack(( 1, 2, np.zeros((nb_var - param["nbVarPos"],1)) ))
#))
#Q = np.zeros(( nb_var * param["nbData"] , nb_var * param["nbData"] ))   # Task precision
#Q[-nb_var:,-nb_var:] = np.identity(nb_var)

tl = np.linspace(0, param["nbData"], param["nbPoints"]+1)
tl = np.rint(tl[1:]).astype(np.int64) - 1 
idx_slices = [slice(i,i+nb_var,1) for i in (tl* nb_var)]
param["muQ"] = np.zeros((nb_var * param["nbData"], 1))
Q = np.zeros((nb_var * param["nbData"] , nb_var * param["nbData"]))   # Task precision

for slice_t in idx_slices:
    x_t = np.zeros(( nb_var,1 ))
    x_t[:param["nbVarPos"]] = np.random.rand(param["nbVarPos"],1) * 5
    param["muQ"][slice_t] = x_t
    Q[slice_t,slice_t] = np.diag(np.hstack((np.ones(param["nbVarPos"]), np.zeros(nb_var - param["nbVarPos"]))))


# Dynamical System settings (discrete)
# =====================================

A1d = np.zeros((param["nbDeriv"],param["nbDeriv"]))
B1d = np.zeros((param["nbDeriv"],1))

for i in range(param["nbDeriv"]):
    A1d += np.diag( np.ones(param["nbDeriv"]-i) ,i ) * param["dt"]**i * 1/factorial(i)
    B1d[param["nbDeriv"]-i-1] = param["dt"]**(i+1) * 1/factorial(i+1)

# x_t+1 = Ax_t + B
A = np.kron(A1d,np.identity(param["nbVarPos"]))
B = np.kron(B1d,np.identity(param["nbVarPos"]))

# Build Sx and Su transfer matrices
# x = Sux+Sx
Su = np.zeros((nb_var*param["nbData"],param["nbVarPos"] * (param["nbData"]-1))) 
Sx = np.kron(np.ones((param["nbData"],1)),np.eye(nb_var,nb_var))

M = B
for i in range(1,param["nbData"]):
    Sx[i*nb_var:param["nbData"]*nb_var,:] = np.dot(Sx[i*nb_var:param["nbData"]*nb_var,:], A)
    Su[nb_var*i:nb_var*i+M.shape[0],0:M.shape[1]] = M
    M = np.hstack((np.dot(A,M),B)) # [0,nb_state_var-1]

# Batch LQR Reproduction
# =====================================
x0 = np.zeros((nb_var,1))
u_hat = np.linalg.inv(Su.T @ Q @ Su + R) @ Su.T @ Q @ (param["muQ"] - Sx @ x0)
x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1,nb_var))


# Plotting
# =========
plt.figure()
plt.title("2D Trajectory")
plt.scatter(x_hat[0,0],x_hat[0,1],c='black',s=100)
for slice_t in idx_slices:
    plt.scatter(param["muQ"][slice_t][0],param["muQ"][slice_t][1],c='red',s=100)   
plt.plot(x_hat[:,0] , x_hat[:,1], c='black')
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

fig,axs = plt.subplots(2,1)
for i,t in enumerate(tl):
    axs[0].scatter(t,param["muQ"][idx_slices[i]][0],c='red')
axs[0].plot(x_hat[:,0], c='black')
axs[0].set_ylabel("$x_1$")
axs[0].set_xticks([0,param["nbData"]])
axs[0].set_xticklabels(["0","T"])

for i,t in enumerate(tl):
    axs[1].scatter(t,param["muQ"][idx_slices[i]][1],c='red')
axs[1].plot(x_hat[:,1], c='black')
axs[1].set_ylabel("$x_2$")
axs[1].set_xlabel("$t$")
axs[1].set_xticks([0,param["nbData"]])
axs[1].set_xticklabels(["0","T"])

plt.show()
