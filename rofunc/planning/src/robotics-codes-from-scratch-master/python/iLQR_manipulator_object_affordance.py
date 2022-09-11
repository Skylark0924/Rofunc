'''
    Batch iLQR applied to an object affordance planning problem with a planar manipulator, by considering object boundaries.

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

import copy
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
    global param
    x = x.T
    A = np.tril(np.ones([param["nbVarX"],param["nbVarX"]]))
    f = np.vstack((param["linkLengths"] @ np.cos(A @ x), 
                   param["linkLengths"] @ np.sin(A @ x), 
                   np.mod(np.sum(x,0)+np.pi, 2*np.pi) - np.pi)) #x1,x2,o (orientation as single Euler angle for planar robot)
    return f.T

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

def jkin(x):
    global param
    T = np.tril( np.ones((len(x),len(x))) )
    J = np.vstack((
        -np.sin(T@x).T @ np.diag(param["linkLengths"]) @ T,
        np.cos(T@x).T @ np.diag(param["linkLengths"]) @ T,
        np.ones(len(x))
    ))
    return J

# Cost and gradient
def f_reach( x , bounding_boxes = True ):
    global param

    f = logmap(fkin(x),param["mu"])
    J = np.zeros(( len(x) * param["nbVarF"] , len(x) * param["nbVarX"] ))

    for t in range(x.shape[0]):
        f[t,:2] = param["A"][t].T @ f[t,:2] # Object oriented fk
        Jtmp = jkin(x[t])
        Jtmp[:2] = param["A"][t].T @ Jtmp[:2] # Object centered jacobian

        if bounding_boxes:
            for i in range(2):
                if abs(f[t,i]) < param["sizeObj"][i]:
                    f[t,i] = 0
                    Jtmp[i]=0
                else:
                    f[t,i] -=  np.sign(f[t,i]) * param["sizeObj"][i]
        J[ t*param["nbVarF"]:(t+1)*param["nbVarF"] , t*param["nbVarX"]:(t+1)*param["nbVarX"]] = Jtmp
    return f,J

# General param parameters
# ===============================
param = {
    "dt":1e-2,
    "nbData":50,
    "nbIter":50,
    "nbPoints":2,
    "nbVarX":3,
    "nbVarU":3,
    "nbVarF":3,
    "linkLengths":[3,2,1],
    "sizeObj":[.2,.3],
    "tracking_term":1e0,
    "r":1e-6
}

# Task parameters
# ===============================

# Targets
param["mu"] = np.asarray([
    [2,1,-np.pi/6],				# x , y , orientation
    [3,2,-np.pi/3]
])

# Transformation matrices
param["A"] = np.zeros( (param["nbPoints"],2,2) )
for i in range(param["nbPoints"]):
    orn_t = param["mu"][i,-1]
    param["A"][i,:,:] = np.asarray([
        [np.cos(orn_t) , -np.sin(orn_t)],
        [np.sin(orn_t) , np.cos(orn_t)]
    ])

# Regularization matrix
R = np.identity( (param["nbData"]-1) * param["nbVarU"] ) * param["r"]

# Precision matrix
Q = np.identity( param["nbVarF"]  * param["nbPoints"]) * param["tracking_term"]
Qc = copy.deepcopy(Q) # Object affordance constraint matrix

# Constraining the offset of the first and second viapoint to be correlated
Qc[:2,3:5] = -np.identity(2) *1e0
Qc[3:5,:2] = -np.identity(2) *1e0

# System parameters
# ===============================

# Time occurence of viapoints
tl = np.linspace(0,param["nbData"],param["nbPoints"]+1)
tl = np.rint(tl[1:]).astype(np.int64)-1
idx = np.array([ i + np.arange(0,param["nbVarX"],1) for i in (tl* param["nbVarX"])]) 

u = np.zeros( param["nbVarU"] * (param["nbData"]-1) ) # Initial control command
x0 = np.array( [3*np.pi/4 , -np.pi/2 , - np.pi/4] ) # Initial state (in joint space)

# Transfer matrices (for linear system as single integrator)
Su0 = np.vstack([np.zeros((param["nbVarX"], param["nbVarX"]*(param["nbData"]-1))), 
      np.tril(np.kron(np.ones((param["nbData"]-1, param["nbData"]-1)), np.eye(param["nbVarX"])*param["dt"]))]) 
Sx0 = np.kron( np.ones(param["nbData"]) , np.identity(param["nbVarX"]) ).T
Su = Su0[idx.flatten()] # We remove the lines that are out of interest

# Solving iLQR
# ===============================

for i in range( param["nbIter"] ):
    x = Su0 @ u + Sx0 @ x0
    x = x.reshape( (  param["nbData"] , param["nbVarX"]) )

    f, J = f_reach(x[tl])
    fc , Jc = f_reach(x[tl],False)
    du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + Su.T @ Jc.T @ Qc @ Jc @ Su + R) @ (-Su.T @ J.T @ Q @ f.flatten() - Su.T @ Jc.T @ Qc @ fc.flatten() - u * param["r"])

    # Perform line search
    alpha = 1
    cost0 = f.flatten() @ Q @ f.flatten() + fc.flatten() @ Qc @ fc.flatten() + np.linalg.norm(u) * param["r"]
    
    while True:
        utmp = u + du * alpha
        xtmp = Su0 @ utmp + Sx0 @ x0
        xtmp = xtmp.reshape( (  param["nbData"] , param["nbVarX"]) )
        ftmp, _ = f_reach(xtmp[tl])
        fctmp, _ = f_reach(xtmp[tl],False)
        cost = ftmp.flatten() @ Q @ ftmp.flatten() + fctmp.flatten() @ Qc @ fctmp.flatten() + np.linalg.norm(utmp) * param["r"]
        
        if cost < cost0 or alpha < 1e-3:
            u = utmp
            print("Iteration {}, cost: {}, alpha: {}".format(i,cost,alpha))
            break

        alpha /=2

# Ploting
# ===============================

plt.figure()
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

# Get points of interest
f = fkin(x)
f00 = fkin0(x[0])
f10 = fkin0(x[tl[0]])
fT0 = fkin0(x[tl[1]])

plt.plot( f00[:,0] , f00[:,1],c='black',linewidth=5,alpha=.2)
plt.plot( f10[:,0] , f10[:,1],c='black',linewidth=5,alpha=.6)
plt.plot( fT0[:,0] , fT0[:,1],c='black',linewidth=5,alpha=1)

plt.plot(f[:,0],f[:,1],c="black",marker="o",markevery=[0]+tl.tolist()) 

# Plot bounding box or via-points
ax = plt.gca()
color_map = ["deepskyblue","darkorange"]
for i in range(param["nbPoints"]):

    rect_origin = param["mu"][i,:2] - param["A"][i]@np.array(param["sizeObj"])
    rect_orn = param["mu"][i,-1]

    rect = patches.Rectangle(rect_origin,param["sizeObj"][0]*2,param["sizeObj"][1]*2,np.degrees(rect_orn),color=color_map[i])
    ax.add_patch(rect)

plt.show()
