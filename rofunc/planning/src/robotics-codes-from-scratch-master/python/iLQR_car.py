'''
	  iLQR applied on a car parking problem

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

# Parameters
# ===============================
param = {
  "car_length":0.2,
	"dt":1e-2,
	"x_d":np.array([2, 1, np.pi]),
	"precision":1e3,
	"r":1e-3,
	"horizon":1.0,
	"nbIter":30,
}

param["nbSteps"] = int(param["horizon"] / param["dt"])
param["nbDoF"] = 5 # set 4 for velocity control, 5 for acceleration control

##### Velocity Control functions
class VelCar():
  def __init__(self, l, dt):
    self.l = l
    self.dt = dt
    self.nbDoF = 4
  
  # Dynamics functions
  # x = [x_r, y_r, theta, steering_angle]
  # u = [v, dsteering_angle]
  def step(self, x, u):
    x_ = np.zeros(self.nbDoF)
    x_[:2] = x[:2] + self.dt * u[0] * np.array([np.cos(x[2]), np.sin(x[2])])
    x_[2] = x[2] + self.dt * u[0] * np.tan(x[3]) / self.l
    x_[3] = x[3] + self.dt * u[1]
    return x_
    
  # get df(x,u)/dx
  def get_A(self, x, u):
    A = np.eye(param["nbDoF"])
    A[:2,2] = self.dt * u[0] * np.array([-np.sin(x[2]), np.cos(x[2])])
    A[2,3] = self.dt * u[0] * (1+np.tan(x[3])*np.tan(x[3])) / self.l
    return A

  # get df(x,u)/du
  def get_B(self, x, u):
    B = np.zeros((self.nbDoF, 2))
    B[:2,0] = self.dt * np.array([np.cos(x[2]), np.sin(x[2])])
    B[2,0] = self.dt * np.tan(x[3]) / self.l
    B[3,1] = self.dt
    return B

##### Acceleration Control functions
class AccCar():
  def __init__(self, l, dt):
    self.l = l
    self.dt = dt
    self.nbDoF = 5

  # Dynamics functions
  # x = [x_r, y_r, theta, v, steering_angle]
  # u = [dv, dsteering_angle]
  def step(self, x, u):
    x_ = np.zeros(self.nbDoF)
    x_[:2] = x[:2] + self.dt * x[3] * np.array([np.cos(x[2]), np.sin(x[2])])
    x_[2] = x[2] + self.dt * x[3] * np.tan(x[4]) / self.l
    x_[3:] = x[3:] + self.dt * u
    return x_

  # get df(x,u)/dx
  def get_A(self, x, u):
    A = np.eye(self.nbDoF)
    A[:2,2] = self.dt * x[3] * np.array([-np.sin(x[2]), np.cos(x[2])])
    A[:2,3] = self.dt * np.array([np.cos(x[2]), np.sin(x[2])])
    A[2,3] = self.dt * np.tan(x[4]) / self.l
    A[2,4] = self.dt * x[3] * (1+np.tan(x[4])*np.tan(x[4])) / self.l
    return A

  # get df(x,u)/du
  def get_B(self, x, u):
    B = np.zeros((self.nbDoF, 2))
    B[-2:] = self.dt * np.eye(2)
    return B

# Perform the forward pass
def f_reach(param, model, x_init, Us):
  Xs = np.zeros((param["nbSteps"], param["nbDoF"]))
  As = np.zeros((param["nbSteps"]-1, param["nbDoF"], param["nbDoF"]))
  Bs = np.zeros((param["nbSteps"]-1, param["nbDoF"], 2))

  Xs[0] = x_init

  for i in range(param["nbSteps"]-1):
    As[i] = model.get_A(Xs[i], Us[i])
    Bs[i] = model.get_B(Xs[i], Us[i])

    Xs[i+1] = model.step(Xs[i], Us[i])
  return Xs, get_system_matrix(param, As, Bs)

# Build Su
def get_system_matrix(param, As, Bs):
  Su = np.zeros((param["nbDoF"]*param["nbSteps"], 2*(param["nbSteps"]-1)))
  for j in range(param["nbSteps"]-1):
    Su[(j+1)*param["nbDoF"]:(j+2)*param["nbDoF"], j*2:(j+1)*2] = Bs[j]
    for i in range(param["nbSteps"]-2-j):
      Su[(j+2+i)*param["nbDoF"]:(j+3+i)*param["nbDoF"], j*2:(j+1)*2] = As[i+j+1] @ Su[(j+1+i)*param["nbDoF"]:(j+2+i)*param["nbDoF"], j*2:(j+1)*2]
  return Su

# Plot the car at the desired position
def plot_car(param, x, alpha=1.0,color='black',label=None):
  w = param["car_length"]/2.
  x_rl = x[:2] + 0.5*w*np.array([-np.sin(x[2]), np.cos(x[2])])
  x_rr = x[:2] - 0.5*w*np.array([-np.sin(x[2]), np.cos(x[2])])
  x_fl = x_rl + param["car_length"] * np.array([np.cos(x[2]), np.sin(x[2])])
  x_fr = x_rr + param["car_length"] * np.array([np.cos(x[2]), np.sin(x[2])])
  x_plot = np.concatenate((x_rl, x_rr, x_fr, x_fl, x_rl))
  
  if label is None:
    plt.plot(x_plot[0::2], x_plot[1::2], linewidth=2, c=color,alpha=alpha)
  else:
    plt.plot(x_plot[0::2], x_plot[1::2], linewidth=2, c=color,alpha=alpha,label=label)
  
# Solving iLQR
# ===============================

if param["nbDoF"] == 4:
  model = VelCar(param["car_length"], param["dt"])
elif param["nbDoF"] == 5:
  model = AccCar(param["car_length"], param["dt"])

Q = np.zeros((param["nbDoF"]*param["nbSteps"],param["nbDoF"]*param["nbSteps"]))
Q[-param["nbDoF"]:,-param["nbDoF"]:] = param["precision"] * np.eye(param["nbDoF"])
R = param["r"] * np.eye(2*(param["nbSteps"]-1))
mu = np.zeros(param["nbDoF"]*param["nbSteps"])
mu[-param["nbDoF"]:-param["nbDoF"]+3] = param["x_d"]
x_init = np.zeros(param["nbDoF"])

U = np.random.normal(0.0, 0.0, (param["nbSteps"]-1, 2))
for k in range(param["nbIter"]):
  X, Su = f_reach(param, model, x_init, U)
  cost0 = (X.flatten() - mu).dot(Q @ (X.flatten() - mu)) + U.flatten().dot(R @ U.flatten())

  du = np.linalg.inv(Su.T @ Q @ Su + R) @ (Su.T @ Q @ (mu - X.flatten()) - R @ U.flatten())
  
  # Perform line search
  alpha = 1
  
  while True:
    Utmp = (U.flatten() + du * alpha).reshape((-1,2))
    Xtmp, _ = f_reach(param, model, x_init, Utmp)
    cost = (Xtmp.flatten() - mu).dot(Q @ (Xtmp.flatten() - mu)) + Utmp.flatten().dot(R @ Utmp.flatten())
    
    if cost < cost0 or alpha < 1e-2:
      U = Utmp
      print("Iteration {}, cost: {}, alpha: {}".format(k,cost,alpha))
      break

    alpha /=2
  
  if np.linalg.norm(alpha * du) < 1e-1:
    break
	
# Plotting
# ===============================

plt.figure()
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

nb_plots = 12
for i in range(nb_plots):
  plot_car(param,X[int(i*len(X)/nb_plots)], 0.4+0.4*i/nb_plots)
plot_car(param,X[-1],label="Trajectory")

print("Final State: ", X[-1])

plot_car(param,param["x_d"],1,'r',label="Goal")
plt.legend()
plt.show()
