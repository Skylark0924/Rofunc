'''
    Forward dynamics in recursive form

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Nadia Hadjmbarek <nadia.hadjmbarek@idiap.ch>, Amirreza Razmjoo <amirreza.razmjoo@idiap.ch>,
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
import matplotlib.animation as animation

# Helper functions
# ===============================

# Forward Kinematics for all joints
def fkin0(x):
    T = np.tril(np.ones([param["nbVarX"],param["nbVarX"]]))
    T2 = np.tril(np.matlib.repmat(param["linkLengths"],len(x),1))
    f = np.vstack(( 
        T2 @ np.cos(T@x),
        T2 @ np.sin(T@x)
    )).T
    f = np.vstack(( 
        np.zeros(2),
        f
    ))
    return f

# Initialization of the plot
def init():
    ax.set_xlim(-np.sum(param["linkLengths"]) - 0.1, np.sum(param["linkLengths"]) + 0.1)
    ax.set_ylim(-np.sum(param["linkLengths"]) - 0.1, np.sum(param["linkLengths"]) + 0.1)
    return ln1, ln2

# Updating the values in the plot
def animate(i):
    f = fkin0(x[0:param["nbVarX"],i])
    ln1.set_data(f[:,0], f[:,1])
    ln2.set_data(f[:,0], f[:,1])
    return ln1, ln2


# General param parameters
# ===============================
param = {
    "dt":1e-2,
    "nbData":500,
    "nbPoints":2,
    "nbVarX":3,
    "nbVarU":3,
    "nbVarF":3,
    "linkLengths":[1,1,1],
    "linkMasses":[1,1,1],
    "damping":1,
    "gravity":9.81,
}

#Auxiliary matrices
# ===============================
l = np.reshape(param["linkLengths"], [1,param["nbVarX"]])
m = np.reshape(param["linkMasses"], [1,param["nbVarX"]])
T = np.tril(np.ones([param["nbVarX"], param["nbVarX"]]))
Tm = np.multiply(np.triu(np.ones([m.shape[1], m.shape[1]])), np.repeat(m, m.shape[1],0)) 


# Initialization
# ===============================
x0 = np.zeros([2*param["nbVarX"], 1]) #initial states
u = np.zeros([param["nbVarU"]*(param["nbData"] - 1), 1]) #Input commands for the whole task.  
x = x0
# Forward Dynamics
for t in range(param["nbData"] - 1):
        
    # Elementwise computation of G, M, and C
    G = np.zeros([param["nbVarX"], 1])
    M = np.zeros([param["nbVarX"], param["nbVarX"]])
    C =  np.zeros([param["nbVarX"], param["nbVarX"]])
    for k in range(param["nbVarX"]):
        G[k,0] = -sum(m[0,k:]) * param["gravity"] * l[0,k] * np.cos(T[k,:] @ x[:param["nbVarX"], t])
        for i in range(param["nbVarX"]):
            S = sum(m[0,k:param["nbVarX"]] * np.heaviside(np.array(range(k, param["nbVarX"])) - i, 1))
            M[k,i] = l[0,k] * l[0,i] * np.cos(T[k,:] @ x[:param["nbVarX"], t] - T[i,:] @ x[:param["nbVarX"], t]) * S
            C[k,i] = -l[0,k] * l[0,i] * np.sin(T[k,:] @ x[:param["nbVarX"], t] - T[i,:] @ x[:param["nbVarX"], t]) * S

    G = T.T @ G
    M = T.T @ M @ T

    # Compute acceleration
    tau = np.reshape(u[(t) * param["nbVarX"]:(t+1)*param["nbVarX"]], [param["nbVarX"], 1])
    inv_M = np.linalg.inv(M)
    ddq =inv_M @ (tau + G + T.T @ C @ (T @ np.reshape(x[param["nbVarX"]:, t],[param["nbVarX"], 1]))**2) - T @ np.reshape(x[param["nbVarX"]:, t],[param["nbVarX"], 1])*param["damping"]

    # compute the next step
    xt = x[:,t].reshape(2 * param["nbVarX"], 1) + np.vstack([x[param["nbVarX"]:, t].reshape(param["nbVarX"], 1),ddq]) * param["dt"]
    x = np.hstack([x, xt])

# Plot
# ===============================
fig, ax = plt.subplots()
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')
xdata, ydata = [], []
ln1, = plt.plot([], [], '-')
ln2, = plt.plot([], [], 'go-', linewidth=2, markersize=5, c="black")
ani = animation.FuncAnimation(fig, animate, x.shape[1], init_func=init, interval = param["dt"] * 1000, blit= True, repeat = False)
plt.show()
