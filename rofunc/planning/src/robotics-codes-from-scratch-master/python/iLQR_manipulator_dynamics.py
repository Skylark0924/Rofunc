'''
    Batch iLQR with computation of the manipulator dynamics

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Amirreza Razmjoo <amirreza.razmjoo@idiap.ch>,
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

# Logarithmic map for R^2 x S^1 manifold
def logmap(f,f0):
    position_error = f[:,:2] - f0[:,:2]
    orientation_error = np.imag(np.log( np.exp( f0[:,-1]*1j ).conj().T * np.exp(f[:,-1]*1j).T )).conj().reshape((-1,1))
    error = np.hstack(( position_error , orientation_error ))
    return error

# Forward kinematics for E-E
def fkin(x):
    global model
    x = x.T
    A = np.tril(np.ones([model["nbVarX"],model["nbVarX"]]))
    f = np.vstack((model["linkLengths"] @ np.cos(A @ x), 
                   model["linkLengths"] @ np.sin(A @ x), 
                   np.mod(np.sum(x,0)+np.pi, 2*np.pi) - np.pi)) #x1,x2,o (orientation as single Euler angle for planar robot)
    return f.T

# Forward Kinematics for all joints
def fkin0(x):
    T = np.tril(np.ones([model["nbVarX"],model["nbVarX"]]))
    T2 = np.tril(np.matlib.repmat(model["linkLengths"],len(x),1))
    f = np.vstack(( 
        T2 @ np.cos(T@x),
        T2 @ np.sin(T@x)
    )).T
    f = np.vstack(( 
        np.zeros(2),
        f
    ))
    return f

def jkin(x):
    global model
    T = np.tril( np.ones((len(x),len(x))) )
    J = np.vstack((
        -np.sin(T@x).T @ np.diag(model["linkLengths"]) @ T,
        np.cos(T@x).T @ np.diag(model["linkLengths"]) @ T,
        np.ones(len(x))
    ))
    return J

# Cost and gradient
def f_reach( x ):
    global model

    f = logmap(fkin(x),model["mu"])
    J = np.zeros(( len(x) * model["nbVarF"] , len(x) * model["nbVarX"] ))

    for t in range(x.shape[0]):
        f[t,:2] = model["A"][t].T @ f[t,:2] # Object oriented fk
        Jtmp = jkin(x[t])
        Jtmp[:2] = model["A"][t].T @ Jtmp[:2] # Object centered jacobian

        if model["useBoundingBox"]:
            for i in range(2):
                if abs(f[t,i]) < model["sizeObj"][i]:
                    f[t,i] = 0
                    Jtmp[i]=0
                else:
                    f[t,i] -=  np.sign(f[t,i]) * model["sizeObj"][i]
        J[ t*model["nbVarF"]:(t+1)*model["nbVarF"] , t*model["nbVarX"]:(t+1)*model["nbVarX"]] = Jtmp
    return f,J

# Forward dynamic to compute
def forward_dynamics(x, u, model):
    
    g = 9.81 #Gravity norm
    kv = 1 #Joints Damping
    l = np.reshape( model["linkLengths"], [1,model["nbVarX"]] )
    m = np.reshape( model["linkMasses"], [1,model["nbVarX"]] )
    dt = model["dt"]
    
    nbDOFs = l.shape[1]
    nbData = int(u.shape[0]/nbDOFs + 1)
    Tm = np.multiply(np.triu(np.ones([nbDOFs,nbDOFs])), np.repeat(m, nbDOFs,0)) 
    T = np.tril(np.ones([nbDOFs, nbDOFs]))
    Su = np.zeros([2 * nbDOFs * nbData, nbDOFs * (nbData - 1)])
    
    #Precomputation of mask (in tensor form)
    S1= np.zeros([nbDOFs, nbDOFs, nbDOFs])
    J_index = np.ones([1, nbDOFs])
    for j in range(nbDOFs):
        J_index[0,:j] = np.zeros([j])
        S1[:,:,j] = np.repeat(J_index @ np.eye(nbDOFs), nbDOFs, 0) - np.transpose(np.repeat(J_index @ np.eye(nbDOFs), nbDOFs, 0))
     
    #Initialization of dM and dC tensors and A21 matrix
    dM = np.zeros([nbDOFs, nbDOFs, nbDOFs])
    dC = np.zeros([nbDOFs, nbDOFs, nbDOFs])
    A21 = np.zeros([nbDOFs, nbDOFs])
    
    
    for t in range(nbData-1):        
        
        # Computation in matrix form of G,M, and C
        G =-np.reshape(np.sum(Tm,1), [nbDOFs,1]) * l.T * np.cos(T @ np.reshape(x[t,0:nbDOFs], [nbDOFs,1])) * g
        G = T.T @ G
        M = (l.T * l) * np.cos(np.reshape(T @ x[t,:nbDOFs], [nbDOFs,1]) - T @ x[t,:nbDOFs]) * (Tm ** .5 @ ((Tm ** .5).T))
        M = T.T @ M @ T 
        C = -(l.T * l) * np.sin(np.reshape(T @ x[t,:nbDOFs], [nbDOFs,1]) - T @ x[t,:nbDOFs]) * (Tm ** .5 @ ((Tm ** .5).T))
        
        
        # Computation in tensor form of derivatives dG,dM, and dC
        dG = np.diagflat(np.reshape(np.sum(Tm,1), [nbDOFs,1]) * l.T * np.sin(T @ np.reshape(x[t,0:nbDOFs], [nbDOFs,1])) * g) @ T
        dM_tmp = (l.T * l) * np.sin(np.reshape(T @ x[t,:nbDOFs], [nbDOFs,1]) - T @ x[t,:nbDOFs]) * (Tm ** .5 @ ((Tm ** .5).T)) 
        
        for j in range(dM.shape[2]):
            dM[:,:,j] = T.T @ (dM_tmp * S1[:,:,j]) @ T
        
        dC_tmp = (l.T * l) * np.cos(np.reshape( T @ x[t,:nbDOFs], [nbDOFs,1]) - T @ x[t,:nbDOFs]) * (Tm ** .5 @ ((Tm ** .5).T)) 
        for j in range(dC.shape[2]):
            dC[:,:,j] = dC_tmp * S1[:,:,j]   
            
        # update pose 
        tau = np.reshape(u[(t) * nbDOFs:(t + 1) * nbDOFs], [nbDOFs, 1])
        inv_M = np.linalg.inv(M)
        ddq =inv_M @ (tau + G + T.T @ C @ (T @ np.reshape(x[t,nbDOFs:], [nbDOFs,1])) ** 2) - T @ np.reshape(x[t,nbDOFs:], [nbDOFs,1]) * kv
        
        # compute local linear systems
        x[t+1,:] = x[t,:] + np.hstack([x[t,nbDOFs:], np.reshape(ddq, [nbDOFs,])]) * dt
        A11 = np.eye(nbDOFs)
        A12 = A11 * dt
        A22 = np.eye(nbDOFs) + (2 * inv_M @ T.T @ C @ np.diagflat(T @ x[t,nbDOFs:]) @ T - T * kv) * dt
        for j in range(nbDOFs):
            A21[:,j] = (-inv_M @ dM[:,:,j] @ inv_M @ (tau + G + T.T @ C @ (T @ np.reshape(x[t,nbDOFs:], [nbDOFs,1])) ** 2) 
                        + np.reshape(inv_M @ T.T @ dG[:,j], [nbDOFs,1]) + inv_M @ T .T @ dC[:,:,j] @ (T @ np.reshape(x[t,nbDOFs:], [nbDOFs,1])) ** 2).flatten()
        A = np.vstack((np.hstack((A11, A12)), np.hstack((A21 * dt, A22))))
        B = np.vstack((np.zeros([nbDOFs, nbDOFs]), inv_M * dt))
        
        # compute transformation matrix
        Su[2 * nbDOFs * (t + 1):2 * nbDOFs * (t + 2),:] = A @ Su[2 * nbDOFs * t:2 * nbDOFs * (t + 1),:]
        Su[2 * nbDOFs * (t + 1):2 * nbDOFs * (t + 2), nbDOFs * t:nbDOFs * (t + 1)] =B
    return x, Su

# General model parameters
# ===============================
model = {
    "dt":1e-2,
    "nbData":50,
    "nbIter":100,
    "nbPoints":2,
    "nbVarX":3,
    "nbVarU":3,
    "nbVarF":3,
    "linkLengths":[2,2,1],
    "linkMasses":[2,3,4],
    "sizeObj":[.2,.3],
    "useBoundingBox":True,
    "r":1e-6
}

# Task parameters
# ===============================

# Targets
model["mu"] = np.asarray([
    [2,1,-np.pi/3],                # x , y , orientation
    [3,2,-np.pi/3]
])

# Transformation matrices
model["A"] = np.zeros( (model["nbPoints"],2,2) )
for i in range(model["nbPoints"]):
    orn_t = model["mu"][i,-1]
    model["A"][i,:,:] = np.asarray([
        [np.cos(orn_t) , -np.sin(orn_t)],
        [np.sin(orn_t) , np.cos(orn_t)]
    ])

# Regularization matrix
R = np.identity( (model["nbData"]-1) * model["nbVarU"] ) * model["r"]

# Precision matrix
Q = np.identity( model["nbVarF"]  * model["nbPoints"])*1e5

# System parameters
# ===============================

# Time occurence of viapoints
tl = np.linspace(0,model["nbData"],model["nbPoints"]+1)
tl = np.rint(tl[1:]).astype(np.int64)-1
idx = np.array([ i + np.arange(0,model["nbVarX"],1) for i in (tl* 2* model["nbVarX"])]) 

u = np.zeros( model["nbVarU"] * (model["nbData"]-1) ) # Initial control command
x0 = np.array( [3 * np.pi/4 , -np.pi/2 , - np.pi/4] ) # Initial position (in joint space)
v0 = np.array([0,0,0])#initial velocity (in joint space)
x = np.zeros([model["nbData"], 2*model["nbVarX"]])
x[0,:model["nbVarX"]] = x0
x[0,model["nbVarX"]:] = v0
# Transfer matrices (for linear system as single integrator)
Su0 = np.vstack([np.zeros((model["nbVarX"], model["nbVarX"]*(model["nbData"]-1))), 
      np.tril(np.kron(np.ones((model["nbData"]-1, model["nbData"]-1)), np.eye(model["nbVarX"])*model["dt"]))]) 
Sx0 = np.kron( np.ones(model["nbData"]) , np.identity(model["nbVarX"]) ).T
 # We remove the lines that are out of interest

# Solving iLQR
# ===============================

for i in range( model["nbIter"] ):
    # system evolution and Transfer matrix (computed from forward dynamics)
    x, Su0 = forward_dynamics(x, u, model)
    Su = Su0[idx.flatten()]

    f, J = f_reach(x[tl,:model["nbVarX"]])
    du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + R) @ (-Su.T @ J.T @ Q @ f.flatten() - u * model["r"])

    # Perform line search
    alpha = 1
    cost0 = f.flatten() @ Q @ f.flatten() + np.linalg.norm(u) * model["r"]
    
    while True:
        utmp = u + du * alpha
        xtmp, _ = forward_dynamics(x, utmp, model)
        ftmp, _ = f_reach(xtmp[tl,:model["nbVarX"]])
        cost = ftmp.flatten() @ Q @ ftmp.flatten() + np.linalg.norm(utmp) * model["r"]
        
        if cost < cost0 or alpha < 1e-3:
            u = utmp
            print("Iteration {}, cost: {}, alpha: {}".format(i,cost,alpha))
            break
        alpha /=2
    if abs(cost-cost0)/cost < 1e-3:
        break
        
        
# Ploting
# ===============================

plt.figure()
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

# Get points of interest
f = fkin(x[:,:model["nbVarX"]])
f00 = fkin0(x[0,:model["nbVarX"]])
fT0 = fkin0(x[-1,:model["nbVarX"]])

plt.plot( f00[:,0] , f00[:,1],c='black',linewidth=5,alpha=.2)
plt.plot( fT0[:,0] , fT0[:,1],c='black',linewidth=5,alpha=.6)

plt.plot(f[:,0],f[:,1],c="black",marker="o",markevery=[0]+tl.tolist()) #,label="Trajectory"

# Plot bounding box or via-points
ax = plt.gca()
color_map = ["deepskyblue","darkorange"]
for i in range(model["nbPoints"]):
    
    if model["useBoundingBox"]:
        rect_origin = model["mu"][i,:2] - model["A"][i]@np.array(model["sizeObj"])
        rect_orn = model["mu"][i,-1]

        rect = patches.Rectangle(rect_origin,model["sizeObj"][0]*2,model["sizeObj"][1]*2,np.degrees(rect_orn),color=color_map[i])
        ax.add_patch(rect)
    else:
        plt.scatter(model["mu"][i,0],model["mu"][i,1],s=100,marker="X",c=color_map[i])

plt.show()
