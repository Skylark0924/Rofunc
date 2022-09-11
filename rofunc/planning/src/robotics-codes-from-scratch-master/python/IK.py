'''
	Inverse kinematics example

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Sylvain Calinon <https://calinon.ch>

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

T = 50 #Number of datapoints
D = 3 #State space dimension (x1,x2,x3)
l = np.array([2, 2, 1]); #Robot links lengths
fh = np.array([-2, 1]) #Desired target for the end-effector
x = np.ones(D) * np.pi / D #Initial robot pose
L = np.tril(np.ones([D,D])) #Transformation matrix

fig.scatter(fh[0], fh[1], color='r', marker='.', s=10**2) #Plot target
for t in range(T):
	f = np.array([L @ np.diag(l) @ np.cos(L @ x), L @ np.diag(l) @ np.sin(L @ x)]) #Forward kinematics (for all articulations, including end-effector)
	J = np.array([-np.sin(L @ x).T @ np.diag(l) @ L, np.cos(L @ x).T @ np.diag(l) @ L]) #Jacobian (for end-effector)
	x += np.linalg.pinv(J) @ (fh - f[:,-1]) * .1 #Update state 
	f = np.concatenate((np.zeros([2,1]), f), axis=1) #Add robot base (for plotting)
	fig.plot(f[0,:], f[1,:], color='k', linewidth=1) #Plot robot

fig.axis('off')
fig.axis('equal')
fig.show()
