'''
    Linear quadratic tracking (LQT) with control primitives applied to a trajectory
    tracking task, with a formulation similar to dynamical movement primitives (DMP),
    by using the least squares formulation of recursive LQR on an augmented state space

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Sylvain Calinon <https://calinon.ch>
    Written by Boyang Ti <https://www.idiap.ch/~bti/>

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
import os 
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from math import factorial

import scipy.io as scio
from scipy.interpolate import interp1d
from scipy.linalg import block_diag

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
# Parameters
class Param:
    def __init__(self):
        self.dt = 1e-2  # Time step length
        self.nbData = 100  # Number of datapoints
        self.nbSamples = 10  # Number of generated trajectory samples
        self.nbVarU = 2  # Control space dimension (dx1,dx2,dx3)
        self.nbFct = 9  # Number of basis function
        self.nbDeriv = 3  # Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
        self.r = 1e-9  # Control weight term
        self.basisName = "RBF"  # can be PIECEWEISE, RBF, BERNSTEIN, FOURIER


# Main program
# ===============================
param = Param()


nb_var = param.nbVarU * param.nbDeriv  # Dimension of state vector
param.nbVar = nb_var  # Dimension of state vector
nb_varX = nb_var + 1
param.nbVarX = nb_varX  # Augmented state space


# Dynamical System settings (for augmented state space)
# =====================================
A1d = np.zeros(param.nbDeriv)
for i in range(param.nbDeriv):
    A1d = A1d + np.diag(np.ones((1, param.nbDeriv - i)).flatten(), i) * param.dt ** i * 1 / factorial(i)  # Discrete 1D

B1d = np.zeros((param.nbDeriv, 1))
for i in range(param.nbDeriv):
    B1d[param.nbDeriv - 1 - i] = param.dt ** (i + 1) * 1 / factorial(i + 1)  # Discrete 1D

A0 = np.kron(A1d, np.eye(param.nbVarU))  # Discrete nD
B0 = np.kron(B1d, np.eye(param.nbVarU))  # Discrete nD

A = np.vstack((np.hstack((A0, np.zeros((param.nbVar, 1)))), np.hstack((np.zeros((param.nbVar)), 1)).reshape(1, -1)))  # Augmented A (homogeneous)
B = np.vstack((B0, np.zeros((1, param.nbVarU))))  # Augmented B (homogeneous)

# Build Sx and Su transfer matrices (for augmented state space)
Sx = np.kron(np.ones((param.nbData, 1)), np.eye(param.nbVarX, param.nbVarX))
Su = np.zeros((param.nbVarX*param.nbData, param.nbVarU * (param.nbData-1)))  # It's maybe n-1 not sure
M = B
for i in range(1, param.nbData):
    Sx[i*param.nbVarX:param.nbData*param.nbVarX, :] = np.dot(Sx[i*param.nbVarX:param.nbData*param.nbVarX, :], A)
    Su[param.nbVarX*i:param.nbVarX*i+M.shape[0], 0:M.shape[1]] = M
    M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]


# Building basis funcitons
# =========================
functions = {
    "PIECEWEISE": build_phi_piecewise,
    "RBF": build_phi_rbf,
    "BERNSTEIN": build_phi_bernstein,
    "FOURIER": build_phi_fourier
}
phi = functions[param.basisName](param.nbData-1, param.nbFct)

# Application of basis functions to multidimensional control commands
PSI = np.kron(phi, np.identity(param.nbVarU))

# Task description
# =====================================

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DATASET_LETTER_FILE = "../data/2Dletters/S.npy"
x = np.load(str(Path(FILE_PATH, DATASET_LETTER_FILE)))[0,:,:2].T

f_pos = interp1d(np.linspace(0, np.size(x, 1)-1, np.size(x, 1), dtype=int), x, kind='cubic')
MuPos = f_pos(np.linspace(0, np.size(x, 1)-1, param.nbData))  # Position
MuVel = np.gradient(MuPos)[1] / param.dt
MuAcc = np.gradient(MuVel)[1] / param.dt

# Position, velocity and acceleration profiles as references
Mu = np.vstack((MuPos, MuVel, MuAcc, np.zeros((param.nbVar-3*param.nbVarU, param.nbData))))

# Task setting (tracking of acceleration profile and reaching of an end-point)
Q = np.kron(np.identity(param.nbData), np.diag(np.concatenate((np.zeros((param.nbVarU*2)), np.ones(param.nbVarU)*1e-6))))

Q[-1-param.nbVar+1:-1-param.nbVar+2*param.nbVarU+1, -1-param.nbVar+1:-1-param.nbVar+2*param.nbVarU+1] = np.identity(2*param.nbVarU) * 1e0

# Weighting matrices in augmented state form
Qm = np.zeros((param.nbVarX*param.nbData, param.nbVarX*param.nbData))

for t in range(param.nbData):
    id0 = np.linspace(0, param.nbVar-1, param.nbVar, dtype=int) + t * param.nbVar
    id = np.linspace(0, param.nbVarX-1, param.nbVarX, dtype=int) + t * param.nbVarX
    Qm[id[0]:id[-1]+1, id[0]:id[-1]+1] = np.vstack((np.hstack((np.identity(param.nbVar), np.zeros((param.nbVar, 1)))), np.append(-Mu[:, t].reshape(1, -1), 1))) \
    @ block_diag((Q[id0[0]:id0[-1]+1, id0[0]:id0[-1]+1]), 1) @ np.vstack((np.hstack((np.identity(param.nbVar), -Mu[:, t].reshape(-1, 1))), np.append(np.zeros((1, param.nbVar)), 1)))

Rm = np.identity((param.nbData-1)*param.nbVarU) * param.r


# Least squares formulation of recursive LQR with an augmented state space and and control primitives

xn = np.vstack((np.array([[3], [-0.5]]), np.zeros((param.nbVarX-param.nbVarU, 1))))  # Simulated noise on state

W = np.linalg.inv(PSI.T @ Su.T @ Qm @ Su @ PSI + PSI.T @ Rm @ PSI) @ PSI.T @ Su.T @ Qm @ Sx
F = PSI @ W  # F with control primitives

# Reproduction with feedback controller on augmented state space (with CP)
Ka = np.empty((param.nbData-1, param.nbVarU, param.nbVarX))
Ka[0, :, :] = F[0:param.nbVarU, :]
P = np.identity(param.nbVarX)
for t in range(param.nbData-2):
    id = t * param.nbVarU + np.linspace(2, param.nbVarU+1, param.nbVarU, dtype=int)
    P = P @ np.linalg.pinv(A - B @ Ka[t, :, :])
    Ka[t+1, :, :] = F[id, :] @ P

r = np.empty((2, param.nbVarX, param.nbData-1))
for n in range(2):
    x = np.append(Mu[:, 0] + np.append(np.array([2, 1]), np.zeros(param.nbVar-2)), 1).reshape(-1, 1)
    for t in range(param.nbData-1):
        # Feedback control on augmented state (resulting in feedback and feedforward terms on state)
        u = -Ka[t, :, :] @ x
        x = A @ x + B @ u  # Update of state vector
        if t == 24 and n == 1:
            x = x + xn  # Simulated noise on the state
        r[n, :, t] = x.flatten()  # State

# Plot 2D
plt.figure()
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

plt.plot(Mu[0, :], Mu[1, :], c='blue', linestyle='-', linewidth=2)
plt.scatter(Mu[0, -1], Mu[1, -1], c='red', s=100)
plt.scatter(r[0, 0, 0], r[0, 1, 0], c='black', s=50)
plt.plot(r[0, 0, :], r[0, 1, :], c='black', linestyle=':', linewidth=2)
plt.plot(r[1, 0, :], r[1, 1, :], c='black', linestyle='-', linewidth=2)
plt.scatter(r[1, 0, 23:25], r[1, 1, 23:25], c='green', s=30)

plt.show()