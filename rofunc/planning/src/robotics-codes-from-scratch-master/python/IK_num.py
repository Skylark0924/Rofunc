'''
	Inverse kinematics with numerical computation for a planar manipulator

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Jérémy Maceiras <jeremy.maceiras@idiap.ch>,Sylvain Calinon <https://calinon.ch>

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
import matplotlib.pyplot as fig
import copy

# Forward kinematics (for end-effector)
def fkin(x,l):
    L = np.tril(np.ones([len(x),len(x)])) #Transformation matrix
    f = np.array([l.T  @ np.cos(L @ x),l.T @ np.sin(L@x)])
    return f

# Forward kinematics (for all articulation)
def fkin0(x,l):
    L = np.tril(np.ones([len(x),len(x)])) #Transformation matrix
    f = np.array([L @ np.diag(l) @ np.cos(L @ x), L @ np.diag(l) @ np.sin(L @ x)]) #Forward kinematics (for all articulations, including end-effector)
    
    return f

# Jacobian with numerical computation
def jkin(x,l):
    eps = 1e-6
    D = len(x)

    # Matrix computation
    X = np.tile(x.reshape((-1,1)),[1,D])
    F1 = fkin(X,l)
    F2 = fkin(X+np.identity(D)*eps,l)
    J = (F2-F1) / eps

    # # For loop computation
    # J = np.zeros((2,len(x)))
    # f = fkin(x,l)
    # for i in range(len(x)):
    #     xtmp = copy.deepcopy(x)
    #     xtmp[i] += eps 
    #     ftmp = fkin(xtmp,l)
    #     J[:,i] = (ftmp[:,-1] - f[:,-1])/eps

    return J

T = 50 #Number of datapoints
D = 3 #State space dimension (x1,x2,x3)
l = np.array([2, 2, 1]); #Robot links lengths
fh = np.array([-2, 1]) #Desired target for the end-effector
x = np.ones(D) * np.pi / D #Initial robot pose

fig.scatter(fh[0], fh[1], color='r', marker='.', s=10**2) #Plot target
for t in range(T):
	
    J = jkin(x,l)
    f = fkin0(x,l)

    x += np.linalg.pinv(J) @ (fh - f[:,-1]) * .1 #Update state 
    f = np.concatenate((np.zeros([2,1]), f), axis=1) #Add robot base (for plotting)
    fig.plot(f[0,:], f[1,:], color='k', linewidth=1) #Plot robot

fig.axis('off')
fig.axis('equal')
fig.show()
