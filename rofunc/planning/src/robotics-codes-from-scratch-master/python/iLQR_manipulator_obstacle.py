'''
    Batch iLQR with obstacles avoidance applied on a manipulator example

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
def fkin(x,ls):
	x = x.T
	A = np.tril(np.ones([len(ls),len(ls)]))
	f = np.vstack((ls @ np.cos(A @ x), 
				   ls @ np.sin(A @ x), 
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

def f_avoid(x):
    global param
    f=[]
    idx=[]
    J=np.zeros((0,0))
    
    for i in range(param["nbObstacles"]):
        for j in range(param["nbVarU"]):
            for t in range(param["nbData"]):
                
                f_x = fkin(x[t,:(j+1)],param["linkLengths"][:(j+1)]).flatten()
                e = param["U_obs"][i].T @ (f_x[:2] - param["Obs"][i][:2])
                ftmp = 1 - e.T @ e

                if ftmp > 0:
                    f += [ftmp]
                    J_rbt = np.hstack(( jkin(x[t,:(j+1)] , param["linkLengths"][:(j+1)]), np.zeros(( param["nbVarF"] , param["nbVarU"]-(j+1))) ))
                    J_tmp = (-e.T @ param["U_obs"][i].T @ J_rbt[:2]).reshape((-1,1))
                    #J_tmp = 1*(J_rbt[:2].T @ param["U_obs"][i] @ e).reshape((-1,1))

                    J2 = np.zeros(( J.shape[0] + J_tmp.shape[0] , J.shape[1] + J_tmp.shape[1] ))
                    J2[:J.shape[0],:J.shape[1]] = J
                    J2[-J_tmp.shape[0]:,-J_tmp.shape[1]:] = J_tmp
                    J = J2 # Numpy does not provid a blockdiag function...

                    idx.append( (t-1)*param["nbVarU"] + np.array(range(param["nbVarU"])) )

    f = np.array(f)
    idx = np.array(idx)
    return f,J.T,idx

def jkin(x,ls):
    global param
    T = np.tril( np.ones((len(ls),len(ls))) )
    J = np.vstack((
        -np.sin(T@x).T @ np.diag(ls) @ T,
        np.cos(T@x).T @ np.diag(ls) @ T,
        np.ones(len(ls))
    ))
    return J

# Cost and gradient
def f_reach( x ):
    global param

    f = logmap(fkin(x,param["linkLengths"]),param["mu"])
    J = np.zeros(( len(x) * param["nbVarF"] , len(x) * param["nbVarX"] ))

    for t in range(x.shape[0]):
        Jtmp = jkin(x[t], param["linkLengths"])
        J[ t*param["nbVarF"]:(t+1)*param["nbVarF"] , t*param["nbVarX"]:(t+1)*param["nbVarX"]] = Jtmp
    return f,J

# General param parameters
# ===============================
param = {
"dt":1e-2,
"nbData":50,
"nbIter":20,
"nbPoints":1,
"nbObstacles":2,
"nbVarX":3,
"nbVarU":3,
"nbVarF":3,
"linkLengths":[2,2,1],
"sizeObstacle":[.5,.8],
"Q_track":1e2,
"Q_avoid":1e0,
"r":1e-3
}

# Task parameters
# ===============================

# Targets
param["mu"] = np.asarray([
[3,-1,0] ])				# x , y , orientation


# Obstacles
param["Obs"] = np.array([
[2.8, 2, np.pi/4],        # [x1,x2,o]
[3.5, .5, -np.pi/6]        # [x1,x2,o]
])

param["A_obs"] = np.zeros(( param["nbObstacles"] , 2 , 2 ))
param["S_obs"] = np.zeros(( param["nbObstacles"] , 2 , 2 ))
param["R_obs"] = np.zeros(( param["nbObstacles"] , 2 , 2 ))
param["Q_obs"] = np.zeros(( param["nbObstacles"] , 2 , 2 ))
param["U_obs"] = np.zeros(( param["nbObstacles"] , 2 , 2 )) # Q_obs[t] = U_obs[t].T @ U_obs[t]
for i in range(param["nbObstacles"]):
    orn_t = param["Obs"][i][-1]
    param["A_obs"][i] = np.array([              # Orientation in matrix form
    [ np.cos(orn_t) , -np.sin(orn_t)  ],
    [ np.sin(orn_t) , np.cos(orn_t)]
    ])

    param["S_obs"][i] = param["A_obs"][i] @ np.diag(param["sizeObstacle"])**2 @ param["A_obs"][i].T # Covariance matrix
    param["R_obs"][i] = param["A_obs"][i] @ np.diag(param["sizeObstacle"]) # Square root of covariance matrix

    param["Q_obs"][i] = np.linalg.inv( param["S_obs"][i] ) # Precision matrix
    param["U_obs"][i] = param["A_obs"][i] @ np.diag(1/np.array(param["sizeObstacle"])) # "Square root" of param["Q_obs"][i]


# Regularization matrix
R = np.identity( (param["nbData"]-1) * param["nbVarU"] ) * param["r"]

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

    f, J = f_reach(x[tl])  # Tracking objective
    f2, J2, id2 = f_avoid(x)# Avoidance objective

    if len(id2) > 0: # Numpy does not allow zero sized array as Indices
        Su2 = Su0[id2.flatten()]
        du = np.linalg.inv(Su.T @ J.T @ J @ Su * param["Q_track"] + Su2.T @ J2.T @ J2 @ Su2 * param["Q_avoid"] + R) @ \
            (-Su.T @ J.T @ f.flatten() * param["Q_track"] - Su2.T @ J2.T @ f2.flatten() * param["Q_avoid"] - u * param["r"])
    else: # It means that we have a collision free path
        du = np.linalg.inv(Su.T @ J.T @ J @ Su * param["Q_track"] + R) @ \
            (-Su.T @ J.T @ f.flatten() * param["Q_track"] - u * param["r"])
    # Perform line search
    alpha = 1
    cost0 = np.linalg.norm(f.flatten())**2 * param["Q_track"] + np.linalg.norm(f2.flatten())**2 * param["Q_avoid"] + np.linalg.norm(u) * param["r"]

    while True:
        utmp = u + du * alpha
        xtmp = Su0 @ utmp + Sx0 @ x0
        xtmp = xtmp.reshape( (  param["nbData"] , param["nbVarX"]) )
        ftmp, _ = f_reach(xtmp[tl])
        f2tmp,_,_ = f_avoid(xtmp)
        cost = np.linalg.norm(ftmp.flatten())**2 * param["Q_track"] + np.linalg.norm(f2tmp.flatten())**2 * param["Q_avoid"] + np.linalg.norm(utmp) * param["r"]

        if cost < cost0 or alpha < 1e-3:
            u = utmp
            print("Iteration {}, cost: {}, alpha: {}".format(i,cost,alpha))
            break

        alpha /=2
    
    if np.linalg.norm(alpha * du) < 1e-2: # Early stop condition
       break

# Ploting
# ===============================

plt.figure()
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

# Get points of interest
f = fkin(x,param["linkLengths"])

for i in range(param["nbVarX"]-1):
    fi = fkin( x[:,:i+1] , param["linkLengths"][:i+1] )
    plt.plot(fi[:,0],fi[:,1],c="black",alpha=.5,linestyle='dashed')

f00 = fkin0(x[0])
fT0 = fkin0(x[-1])

plt.plot( f00[:,0] , f00[:,1],c='black',linewidth=5,alpha=.2)
plt.plot( fT0[:,0] , fT0[:,1],c='black',linewidth=5,alpha=.6)

plt.plot(f[:,0],f[:,1],c="black",marker="o",markevery=[0]+tl.tolist()) #,label="Trajectory"


# Plot bounding box or via-points
ax = plt.gca()
color_map = ["deepskyblue","darkorange"]
for i in range(param["nbPoints"]):
        plt.scatter(param["mu"][i,0],param["mu"][i,1],s=100,marker="X",c=color_map[i])

# Plot obstactles
al = np.linspace(-np.pi,np.pi,50)
ax = plt.gca()
for i in range(param["nbObstacles"]):
    D,V = np.linalg.eig(param["S_obs"][i])
    D = np.diag(D)
    R = np.real(V@np.sqrt(D+0j))
    msh = (R @ np.array([np.cos(al),np.sin(al)])).T + param["Obs"][i][:2]
    p=patches.Polygon(msh,closed=True)
    ax.add_patch(p)

plt.show()
