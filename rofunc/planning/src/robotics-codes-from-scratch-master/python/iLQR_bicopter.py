'''
	  iLQR applied on a bicopter problem

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Julius Jankowski <julius.jankowski@idiap.ch>,
    Jérémy Maceiras <jeremy.maceiras@idiap.ch>,Sylvain Calinon <https://calinon.ch>

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

# Dynamics functions
# x = [x_g, y_g, theta, dx_g, dy_g, dtheta]
# u = [f_l, f_r]
def step(param, x, u):
  x_ = np.zeros(param["nbDoF"])
  x_[3] = x[3] - param["dt"] * np.sin(x[2])*(u[0]+u[1]) / param["mass"]
  x_[4] = x[4] + param["dt"] * (np.cos(x[2]) * (u[0]+u[1])  / param["mass"] - param["g"])
  x_[5] = x[5] + param["dt"] * 6. * (u[1]-u[0]) / (param["mass"]*param["size"]) # I = m*l^2/12
  x_[:3] = x[:3] + param["dt"] * 0.5 * (x[3:] + x_[3:])
  return x_

# Perform the forward pass
def f_reach(param, x_init, Us):
  Xs = np.zeros((param["nbSteps"], param["nbDoF"]))
  As = np.zeros((param["nbSteps"]-1, param["nbDoF"], param["nbDoF"]))
  Bs = np.zeros((param["nbSteps"]-1,param["nbDoF"],2))

  Xs[0] = x_init

  for i in range(param["nbSteps"]-1):
    As[i] = get_A(param, Xs[i], Us[i])
    Bs[i] = get_B(param, Xs[i])

    Xs[i+1] = step(param, Xs[i], Us[i])
  return Xs,get_system_matrix(As,Bs)

# Get df(x,u)/dx
def get_A(param, x, u):
  A = np.eye(param["nbDoF"])
  A[:3,3:] = param["dt"] * np.eye(3)
  A[0:2,2] = -0.5 * param["dt"] * param["dt"] / param["mass"] * (u[0]+u[1]) * np.array([np.cos(x[2]), np.sin(x[2])])
  A[3:5,2] = -param["dt"] / param["mass"] * (u[0]+u[1]) * np.array([np.cos(x[2]), np.sin(x[2])])
  return A

# Get df(x,u)/du
def get_B(param, x):
  B = np.zeros((param["nbDoF"], 2))
  M_inv = np.diag(np.array([1./param["mass"], 1./param["mass"], 12./(param["mass"]*param["size"]*param["size"])]))
  G = np.array([[-np.sin(x[2]), -np.sin(x[2])], [np.cos(x[2]), np.cos(x[2])], [-0.5*param["size"], 0.5*param["size"]]])
  B[:3] = 0.5 * param["dt"] * param["dt"] * M_inv @ G
  B[3:] = param["dt"] * M_inv @ G
  return B

# Build Su
def get_system_matrix(As,Bs):
  Su = np.zeros((param["nbDoF"]*param["nbSteps"], 2*(param["nbSteps"]-1)))
  for j in range(param["nbSteps"]-1):
    Su[(j+1)*param["nbDoF"]:(j+2)*param["nbDoF"], j*2:(j+1)*2] = Bs[j]
    for i in range(param["nbSteps"]-2-j):
      Su[(j+2+i)*param["nbDoF"]:(j+3+i)*param["nbDoF"], j*2:(j+1)*2] = As[i+j+1] @ Su[(j+1+i)*param["nbDoF"]:(j+2+i)*param["nbDoF"], j*2:(j+1)*2]
  return Su

# Compute the cost
def get_cost(X, U):
  return (X.flatten() - mu).dot(Q @ (X.flatten() - mu)) + U.flatten().dot(R @ U.flatten())
	
# Plot the bicopter at the desired position
def plot_bicopter(param, x, alpha=1.0):
  offset = 0.5 * param["size"] * np.array([np.cos(x[2]), np.sin(x[2])])
  xl = x[:2] - offset
  xr = x[:2] + offset
  plt.plot([xl[0], xr[0]], [xl[1], xr[1]], c='black', linewidth=2, alpha=alpha)

# Parameters
# ===============================
param = {
  "size":0.2, # Size of the bicopter
  "mass":1.0, # Mass of the bicopter
  "g":9.81, # Gravity
	"dt":1e-2,
	"x_d":np.array([-2, 1, 0 ]), # Desired_position
	"precision":1e3,
	"r":1e-3,
	"horizon":1.0,
	"nbIter":30,
}

param["nbSteps"] = int(param["horizon"] / param["dt"])
param["nbDoF"] = 6
  
# Solving iLQR
# ===============================

Q = np.zeros((param["nbDoF"]*param["nbSteps"],param["nbDoF"]*param["nbSteps"]))
Q[-param["nbDoF"]:,-param["nbDoF"]:] = param["precision"] * np.eye(param["nbDoF"])
R = param["r"] * np.eye(2*(param["nbSteps"]-1))

mu = np.zeros(param["nbDoF"]*param["nbSteps"])
mu[-param["nbDoF"]:-param["nbDoF"]+3] = param["x_d"]
x_init = np.zeros(param["nbDoF"])

U = np.random.normal(0.5*param["mass"]*param["g"], 0.001, (param["nbSteps"]-1, 2))
for k in range(param["nbIter"]):
  X, Su = f_reach(param,x_init, U)    
  cost0 = get_cost(X,U)
  du = np.linalg.inv(Su.T @ Q @ Su + R) @ (Su.T @ Q @ (mu - X.flatten()) - R @ U.flatten())
  
  # Perform line search
  alpha = 1
  
  while True:
    Utmp = (U.flatten() + du * alpha).reshape((-1,2))
    Xtmp, _ = f_reach(param,x_init,Utmp)
    cost = get_cost(Xtmp,Utmp)
    
    if cost < cost0 or alpha < 1e-3:
      U = Utmp
      print("Iteration {}, cost: {}, alpha: {}".format(k,cost,alpha))
      break

    alpha /=2

# Ploting
# =========

plt.figure()
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

plt.plot(X[:,0],X[:,1], c='black')

nb_plots = 15
for i in range(nb_plots):
  plot_bicopter(param, X[int(i*len(X)/nb_plots)], 0.4+0.4*i/nb_plots)
  plt.scatter( X[int(i*len(X)/nb_plots)][0] , X[int(i*len(X)/nb_plots)][1] , marker='.', s=200, c='black', alpha=0.4+0.4*i/nb_plots  )

plot_bicopter(param, X[-1])
plt.scatter(X[-1][0],X[-1][1],c='black')

plt.scatter(param["x_d"][0], param["x_d"][1], color='r', marker='.', s=200,label="Desired pose")
plt.legend()

plt.show()
