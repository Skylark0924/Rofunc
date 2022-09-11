'''
    iLQR applied to a planar manipulator for a viapoints task (batch formulation)

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Jeremy Maceiras <jeremy.maceiras@idiap.ch> and
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
import matplotlib.patches as patches

# Helper functions
# ===============================

# Logarithmic map for R^2 x S^1 manifold
def logmap(f, f0):
	position_error = f[:2,:] - f0[:2,:]
	orientation_error = np.imag(np.log(np.exp(f0[-1,:]*1j).conj().T * np.exp(f[-1,:]*1j).T)).conj()
	error = np.vstack([position_error, orientation_error])
	return error

# Forward kinematics for end-effector (in robot coordinate system)
def fkin(x, param):
	L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
	f = np.vstack([
		param.l @ np.cos(L @ x),
		param.l @ np.sin(L @ x),
		np.mod(np.sum(x,0)+np.pi, 2*np.pi) - np.pi
	]) # f1,f2,f3, where f3 is the orientation (single Euler angle for planar robot)
	return f

# Forward kinematics for end-effector (in robot coordinate system)
def fkin0(x, param): 
	L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
	f = np.vstack([
		L @ np.diag(param.l) @ np.cos(L @ x),
		L @ np.diag(param.l) @ np.sin(L @ x)
	])
	f = np.hstack([np.zeros([2,1]), f])
	return f

# Jacobian with analytical computation (for single time step)
def Jkin(x, param):
	L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
	J = np.vstack([
		-np.sin(L @ x).T @ np.diag(param.l) @ L,
		np.cos(L @ x).T @ np.diag(param.l) @ L,
		np.ones([1,param.nbVarX])
	])
	return J

# Error and Jacobian for a viapoints reaching task (in object coordinate system)
def f_reach(x, param):
	f = logmap(fkin(x, param), param.Mu)
	J = np.zeros([param.nbPoints * param.nbVarF, param.nbPoints * param.nbVarX])
	for t in range(param.nbPoints):
		f[:2,t] = param.A[:,:,t].T @ f[:2,t] # Object oriented forward kinematics
		Jtmp = Jkin(x[:,t], param)
		Jtmp[:2] = param.A[:,:,t].T @ Jtmp[:2] # Object centered Jacobian
		
		if param.useBoundingBox:
			for i in range(2):
				if abs(f[i,t]) < param.sz[i]:
					f[i,t] = 0
					Jtmp[i] = 0
				else:
					f[i,t] -= np.sign(f[i,t]) * param.sz[i]
		
		J[t*param.nbVarF:(t+1)*param.nbVarF, t*param.nbVarX:(t+1)*param.nbVarX] = Jtmp
	return f,J

# Parameters
class Param:
	def __init__(self):
		self.dt = 1e-2 # Time step length
		self.nbData = 50 # Number of datapoints
		self.nbIter = 100 # Maximum number of iterations for iLQR
		self.nbPoints = 2 # Number of viapoints
		self.nbVarX = 3 # State space dimension (x1,x2,x3)
		self.nbVarU = 3 # Control space dimension (dx1,dx2,dx3)
		self.nbVarF = 3 # Objective function dimension (f1,f2,f3, with f3 as orientation)
		self.l = [2,2,1] # Robot links lengths
		self.sz = [.2,.3] # Size of objects
		self.r = 1e-6 # Control weight term
		self.Mu = np.asarray([[2,1,-np.pi/6], [3,2,-np.pi/3]]).T # Viapoints 
		self.A = np.zeros([2,2,self.nbPoints]) # Object orientation matrices
		self.useBoundingBox = True # Consider bounding boxes for reaching cost


# Main program
# ===============================

param = Param();

# Object rotation matrices
for t in range(param.nbPoints):
	orn_t = param.Mu[-1,t]
	param.A[:,:,t] = np.asarray([
		[np.cos(orn_t), -np.sin(orn_t)],
		[np.sin(orn_t), np.cos(orn_t)]
	])

# Precision matrix
Q = np.identity(param.nbVarF * param.nbPoints)

# Control weight matrix
R = np.identity((param.nbData-1) * param.nbVarU) * param.r

# Time occurrence of viapoints
tl = np.linspace(0, param.nbData, param.nbPoints+1)
tl = np.rint(tl[1:]).astype(np.int64) - 1
idx = np.array([i + np.arange(0,param.nbVarX,1) for i in (tl*param.nbVarX)]) 

# Transfer matrices (for linear system as single integrator)
Su0 = np.vstack([
	np.zeros([param.nbVarX, param.nbVarX*(param.nbData-1)]), 
	np.tril(np.kron(np.ones([param.nbData-1, param.nbData-1]), np.eye(param.nbVarX) * param.dt))
]) 
Sx0 = np.kron(np.ones(param.nbData), np.identity(param.nbVarX)).T
Su = Su0[idx.flatten()] # We remove the lines that are out of interest


# iLQR
# ===============================

u = np.zeros(param.nbVarU * (param.nbData-1)) # Initial control command
x0 = np.array([3*np.pi/4, -np.pi/2, -np.pi/4]) # Initial state

for i in range(param.nbIter):
	x = Su0 @ u + Sx0 @ x0 # System evolution
	x = x.reshape([param.nbVarX, param.nbData], order='F')
	f,J = f_reach(x[:,tl], param) # Residuals and Jacobians
	du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + R) @ (-Su.T @ J.T @ Q @ f.flatten('F') - u * param.r) # Gauss-Newton update
	# Estimate step size with backtracking line search method
	alpha = 1
	cost0 = f.flatten('F').T @ Q @ f.flatten('F') + np.linalg.norm(u)**2 * param.r # Cost
	while True:
		utmp = u + du * alpha
		xtmp = Su0 @ utmp + Sx0 @ x0 # System evolution
		xtmp = xtmp.reshape([param.nbVarX, param.nbData], order='F')
		ftmp,_ = f_reach(xtmp[:,tl], param) # Residuals 
		cost = ftmp.flatten('F').T @ Q @ ftmp.flatten('F') + np.linalg.norm(utmp)**2 * param.r # Cost
		if cost < cost0 or alpha < 1e-3:
			u = utmp
			print("Iteration {}, cost: {}".format(i,cost))
			break
		alpha /= 2
	if np.linalg.norm(du * alpha) < 1E-2:
		break # Stop iLQR iterations when solution is reached

# Plots
# ===============================

plt.figure()
plt.axis('off')
plt.gca().set_aspect('equal', adjustable='box')

# Get points of interest
f00 = fkin0(x[:,0], param)
f01 = fkin0(x[:,tl[0]], param)
f02 = fkin0(x[:,tl[1]], param)
f = fkin(x, param)

plt.plot(f00[0,:], f00[1,:], c='black', linewidth=5, alpha=.2)
plt.plot(f01[0,:], f01[1,:], c='black', linewidth=5, alpha=.4)
plt.plot(f02[0,:], f02[1,:], c='black', linewidth=5, alpha=.6)
plt.plot(f[0,:], f[1,:], c='black', marker='o', markevery=[0]+tl.tolist()) 

# Plot bounding box or viapoints
ax = plt.gca()
color_map = ['deepskyblue','darkorange']
for t in range(param.nbPoints):
	if param.useBoundingBox:
		rect_origin = param.Mu[:2,t] - param.A[:,:,t] @ np.array(param.sz)
		rect_orn = param.Mu[-1,t]
		rect = patches.Rectangle(rect_origin, param.sz[0]*2, param.sz[1]*2, np.degrees(rect_orn), color=color_map[t])
		ax.add_patch(rect)
	else:
		plt.scatter(param.Mu[0,t], param.Mu[1,t], s=100, marker='X', c=color_map[t])

plt.show()
