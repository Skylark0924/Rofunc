'''
    Linear Quadratic tracker with control primitives applied on a via-point example

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
import matplotlib.pyplot as plt
from scipy import special
from math import factorial

# Building piecewise constant basis functions
def build_phi_piecewise(nb_data, nb_fct):
    phi = np.kron( np.identity(nb_fct) , np.ones((int(np.ceil(nb_data/nb_fct)),1)) )
    return phi[:nb_data]

# Building radial basis functions (RBFs)
def build_phi_rbf(nb_data, nb_fct):
    t = np.linspace(0,1,nb_data).reshape((-1,1))
    tMu = np.linspace( t[0] , t[-1] , nb_fct )
    phi = np.exp( -1e2 * (t.T - tMu)**2 )
    return phi.T

# Building Bernstein basis functions
def build_phi_bernstein(nb_data, nb_fct):
    t = np.linspace(0,1,nb_data)
    phi = np.zeros((nb_data,nb_fct))
    for i in range(nb_fct):
        phi[:,i] = factorial(nb_fct-1) / (factorial(i) * factorial(nb_fct-1-i)) * (1-t)**(nb_fct-1-i) * t**i
    return phi

# Building Fourier basis functions
def build_phi_fourier(nb_data, nb_fct):

    t = np.linspace(0,1,nb_data).reshape((-1,1))

    # Alternative computation for real and even signal
    k = np.arange(0,nb_fct).reshape((-1,1))
    phi = np.cos( t.T * k * 2 * np.pi )
    return phi.T

# General param parameters
# ===============================

param = {
    "nbData" : 100, # Number of data points
    "nbPoints" : 3, # Number of viapoints
    "nbVarPos" : 2, # Dimension of position data
    "nbDeriv" : 2,  # Number of static and dynamic features (2 -> [x,dx])
    "dt" : 1e-2, # Time step duration
    "nbFct" : 3, # Number of basis function
    "basisName" : "FOURIER", # can be PIECEWEISE, RBF, BERNSTEIN, FOURIER
    "rfactor" : 1e-8 # Control cost
}

nb_var = param["nbVarPos"] * param["nbDeriv"] # Dimension of state vector
R = np.identity( (param["nbData"]-1) * param["nbVarPos"]  ) * param["rfactor"]  # Control cost matrix

tl = np.linspace(0,param["nbData"],param["nbPoints"]+1)
tl = np.rint(tl[1:]).astype(np.int64)-1 
idx_slices = [ slice( i,i+nb_var,1) for i in (tl* nb_var) ]
param["muQ"] = np.zeros(( nb_var * param["nbData"] , 1 ))
Q = np.zeros(( nb_var * param["nbData"] , nb_var * param["nbData"] ))   # Task precision

for slice_t in idx_slices:

    x_t = np.zeros(( nb_var,1 ))
    x_t[:param["nbVarPos"]] = np.random.rand(param["nbVarPos"],1)*5
    
    param["muQ"][slice_t] = x_t
    Q[slice_t,slice_t] = np.diag(  np.hstack(( np.ones(param["nbVarPos"]) , np.zeros( nb_var - param["nbVarPos"])  )) )

# Dynamical System settings (discrete)
# =====================================

A = np.identity( nb_var )
if param["nbDeriv"]==2:
    A[:param["nbVarPos"],-param["nbVarPos"]:] = np.identity(param["nbVarPos"]) * param["dt"]

B = np.zeros(( nb_var , param["nbVarPos"] ))
derivatives = [ param["dt"],param["dt"]**2 /2 ][:param["nbDeriv"]]
for i in range(param["nbDeriv"]):
    B[i*param["nbVarPos"]:(i+1)*param["nbVarPos"]] = np.identity(param["nbVarPos"]) * derivatives[::-1][i]

# Build Sx and Su transfer matrices
Su = np.zeros((nb_var*param["nbData"],param["nbVarPos"] * (param["nbData"]-1))) # It's maybe n-1 not sure
Sx = np.kron(np.ones((param["nbData"],1)),np.eye(nb_var,nb_var))

M = B
for i in range(1,param["nbData"]):
    Sx[i*nb_var:param["nbData"]*nb_var,:] = np.dot(Sx[i*nb_var:param["nbData"]*nb_var,:],A)
    Su[nb_var*i:nb_var*i+M.shape[0],0:M.shape[1]] = M
    M = np.hstack((np.dot(A,M),B)) # [0,nb_state_var-1]

# Building basis funcitons
# =========================

functions = {
    "PIECEWEISE": build_phi_piecewise ,
    "RBF": build_phi_rbf,
    "BERNSTEIN": build_phi_bernstein,
    "FOURIER": build_phi_fourier
}
phi = functions[param["basisName"]](param["nbData"]-1,param["nbFct"])
PSI = np.kron(phi,np.identity(param["nbVarPos"]))

# Batch LQR Reproduction
# =====================================

x0 = np.zeros((nb_var,1))
w_hat = np.linalg.inv(PSI.T @ Su.T @ Q @ Su @ PSI + PSI.T @ R @ PSI) @ PSI.T @ Su.T @ Q @ ( param["muQ"] - Sx @ x0 )
u_hat = PSI @ w_hat
x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1,nb_var))
u_hat = u_hat.reshape((-1,param["nbVarPos"]))

# Plotting
# =========

plt.figure()

plt.title("2D Trajectory")
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

plt.scatter(x_hat[0,0],x_hat[0,1],c='black',s=100)

for slice_t in idx_slices:
    plt.scatter(param["muQ"][slice_t][0],param["muQ"][slice_t][1],c='blue',s=100)

plt.plot( x_hat[:,0] , x_hat[:,1],c='black')

fig,axs = plt.subplots(5,1)

axs[0].plot(x_hat[:,0],c='black')
axs[0].set_ylabel("$x_1$")
axs[0].set_xticks([0,param["nbData"]])
axs[0].set_xticklabels(["0","T"])
for t in tl:
    axs[0].scatter(t,x_hat[t,0],c='blue')

axs[1].plot(x_hat[:,1],c='black')
axs[1].set_ylabel("$x_2$")
axs[1].set_xticks([0,param["nbData"]])
axs[1].set_xticklabels(["0","T"])
for t in tl:
    axs[1].scatter(t,x_hat[t,1],c='blue')

axs[2].plot(u_hat[:,0],c='black')
axs[2].set_ylabel("$u_1$")
axs[2].set_xticks([0,param["nbData"]-1])
axs[2].set_xticklabels(["0","T-1"])

axs[3].plot(u_hat[:,1],c='black')
axs[3].set_ylabel("$u_2$")
axs[3].set_xticks([0,param["nbData"]-1])
axs[3].set_xticklabels(["0","T-1"])

axs[4].set_ylabel("$\phi_k$")
axs[4].set_xticks([0,param["nbData"]-1])
axs[4].set_xticklabels(["0","T-1"])
for i in range(param["nbFct"]):
    axs[4].plot(phi[:,i])
axs[4].set_xlabel("$t$")

plt.show()