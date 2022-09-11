'''
    Movement primitives applied to a 2D trajectory

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
from math import factorial
import os
from pathlib import Path

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
    "nbFct" : 7, # Number of basis functions
    "nbVar" : 2, # Dimension of position data 
    "nbData" : 200, # Number of datapoints in a trajectory
    "basisName": "FOURIER" # PIECEWISE, RBF, BERNSTEIN, FOURIER
}

# Dictionary to call the good method in function of the good basis name.
basis_functions_method = {
    "BERNSTEIN": build_phi_bernstein,
    "PIECEWISE": build_phi_piecewise,
    "RBF": build_phi_rbf,
    "FOURIER": build_phi_fourier
}

# Load handwriting data
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DATASET_LETTER_FILE = "../data/2Dletters/C.npy"
x = np.load(str(Path(FILE_PATH,DATASET_LETTER_FILE)))[0,:param["nbData"],:2]

t = np.linspace(0,1,param["nbData"]) # Time range

# Fourier basis functions require additional symmetrization (mirroring) if the signal is a discrete motion 
if param["basisName"]=="FOURIER":
    x = np.vstack(( x , x[::-1] ))
    param["nbData"] *= 2

x = x.flatten()

# Generate MP with various basis functions
# ============================================
phi = basis_functions_method[param["basisName"]](param["nbData"],param["nbFct"])


psi = np.kron(phi, np.identity(param["nbVar"])) # Transform to multidimensional basis functions
w = np.linalg.solve(psi.T @ psi + np.identity(param["nbVar"] * param["nbFct"]) * 1e-8, psi.T @ x ) # Estimation of superposition weights from data
x_hat = psi @ w # Reconstruction data

# Fourier basis functions require de-symmetrization of the signal after processing (for visualization)
if param["basisName"]=="FOURIER":
    param["nbData"] /= 2
    x = x[:int(param["nbData"]*param["nbVar"])]
    x_hat = x_hat[:int(param["nbData"]*param["nbVar"])] 


# Plotting
# =========
fig,axs = plt.subplots(3,1)
axs[0].plot(x[::2], x[1::2], c='black', label='Original' )
axs[0].plot(x_hat[::2], x_hat[1::2], c='r', label='Reproduced' )
axs[0].axis("off")
axs[0].axis("equal")
axs[0].legend()

axs[1].set_xticks([0,param["nbData"]])
axs[1].set_xticklabels(["0","T"])
for j in range(param["nbFct"]):
    axs[1].plot(phi[:,j])
axs[1].set_ylabel("$\phi_k$")

axs[2].matshow( np.real(psi @ psi.T) )
axs[2].axis("off")
axs[2].set_xticks([])
axs[2].set_yticks([])
axs[2].set_title(" $\Psi \Psi^T$ ",pad=5)

plt.show()
