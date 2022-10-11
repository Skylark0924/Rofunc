LQT (control primitive version)
===============================

Use the control primitive to represent the control command
:math:`\boldsymbol{u}=\boldsymbol \Psi \boldsymbol{w}`

.. math:: \min _{\boldsymbol{w}}(\boldsymbol{x}-\boldsymbol{\mu})^{\top} \boldsymbol{Q}(\boldsymbol{x}-\boldsymbol{\mu})+\boldsymbol{w}^{\top} \boldsymbol\Psi^{\top} \boldsymbol{R} \boldsymbol\Psi \boldsymbol{w}, \quad \text { s.t. } \quad \boldsymbol{x}=\boldsymbol{S}_{\boldsymbol{x}} \boldsymbol{x}_1+\boldsymbol{S}_{\boldsymbol{u}} \boldsymbol \Psi \boldsymbol{w}

the analytic solution is

.. math:: \hat{\boldsymbol w}=\left(\boldsymbol \Psi^{\top} \boldsymbol{S}_u^{\top} \boldsymbol Q \boldsymbol{S}_u \boldsymbol \Psi+\boldsymbol \Psi^{\top} \boldsymbol{R} \boldsymbol \Psi\right)^{-1} \boldsymbol \Psi^{\top} \boldsymbol{S}_u^{\top} \boldsymbol Q\left(\boldsymbol \mu-\boldsymbol S_x \boldsymbol x_1\right)

Define control primitives
-------------------------

.. code:: python

   # Building piecewise constant basis functions
   def build_phi_piecewise(nb_data, nb_fct):
       phi = np.kron(np.identity(nb_fct), np.ones((int(np.ceil(nb_data / nb_fct)), 1)))
       return phi[:nb_data]


   # Building radial basis functions (RBFs)
   def build_phi_rbf(nb_data, nb_fct):
       t = np.linspace(0, 1, nb_data).reshape((-1, 1))
       tMu = np.linspace(t[0], t[-1], nb_fct)
       phi = np.exp(-1e2 * (t.T - tMu) ** 2)
       return phi.T


   # Building Bernstein basis functions
   def build_phi_bernstein(nb_data, nb_fct):
       t = np.linspace(0, 1, nb_data)
       phi = np.zeros((nb_data, nb_fct))
       for i in range(nb_fct):
           phi[:, i] = factorial(nb_fct - 1) / (factorial(i) * factorial(nb_fct - 1 - i)) * (1 - t) ** (
                   nb_fct - 1 - i) * t ** i
       return phi


   # Building Fourier basis functions
   def build_phi_fourier(nb_data, nb_fct):
       t = np.linspace(0, 1, nb_data).reshape((-1, 1))

       # Alternative computation for real and even signal
       k = np.arange(0, nb_fct).reshape((-1, 1))
       phi = np.cos(t.T * k * 2 * np.pi)
       return phi.T

   def define_control_primitive(param):
       functions = {
           "PIECEWEISE": build_phi_piecewise,
           "RBF": build_phi_rbf,
           "BERNSTEIN": build_phi_bernstein,
           "FOURIER": build_phi_fourier
       }
       phi = functions[param["basisName"]](param["nbData"] - 1, param["nbFct"])
       PSI = np.kron(phi, np.identity(param["nbVarPos"]))
       return PSI

.. _define-dynamical-system-1:

Define dynamical system
-----------------------

.. code:: python

   def set_dynamical_system(param: Dict):
       nb_var = param["nb_var"]
       A = np.identity(nb_var)
       if param["nbDeriv"] == 2:
           A[:param["nbVarPos"], -param["nbVarPos"]:] = np.identity(param["nbVarPos"]) * param["dt"]

       B = np.zeros((nb_var, param["nbVarPos"]))
       derivatives = [param["dt"], param["dt"] ** 2 / 2][:param["nbDeriv"]]
       for i in range(param["nbDeriv"]):
           B[i * param["nbVarPos"]:(i + 1) * param["nbVarPos"]] = np.identity(param["nbVarPos"]) * derivatives[::-1][i]

       # Build Sx and Su transfer matrices
       Su = np.zeros((nb_var * param["nbData"], param["nbVarPos"] * (param["nbData"] - 1)))  # It's maybe n-1 not sure
       Sx = np.kron(np.ones((param["nbData"], 1)), np.eye(nb_var, nb_var))

       M = B
       for i in range(1, param["nbData"]):
           Sx[i * nb_var:param["nbData"] * nb_var, :] = np.dot(Sx[i * nb_var:param["nbData"] * nb_var, :], A)
           Su[nb_var * i:nb_var * i + M.shape[0], 0:M.shape[1]] = M
           M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]
       return Su, Sx

Solve the LQT with control primitive 
-------------------------------------

.. math:: \hat{\boldsymbol w}=\left(\boldsymbol \Psi^{\top} \boldsymbol{S}_u^{\top} \boldsymbol Q \boldsymbol{S}_u \boldsymbol \Psi+\boldsymbol \Psi^{\top} \boldsymbol{R} \boldsymbol \Psi\right)^{-1} \boldsymbol \Psi^{\top} \boldsymbol{S}_u^{\top} \boldsymbol Q\left(\boldsymbol \mu-\boldsymbol S_x \boldsymbol x_1\right)\label{3}

.. code:: python

   x0 = start_pose.reshape((14, 1))
   w_hat = np.linalg.inv(PSI.T @ Su.T @ Q @ Su @ PSI + PSI.T @ R @ PSI) @ PSI.T @ Su.T @ Q @ (muQ - Sx @ x0)
   u_hat = PSI @ w_hat
   x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1, param["nb_var"]))
   u_hat = u_hat.reshape((-1, param["nbVarPos"]))

--------------

Reformulate LQT with control primitives via Dynamical movement primitives (DMP) 
--------------------------------------------------------------------------------

Linear quadratic tracking (LQT) with control primitives can be used in a
similar fashion as in DMP, by requesting a target to be reached at the
end of the movement and by requesting the observed acceleration profile
to be tracked, while encoding the control commands as radial basis
functions.

Like the :doc:`Least squares formulation of recursive LQT <./lqt_fb>`,
we can reformat Equ. :math:`\ref{3}` as

.. math:: \hat{\boldsymbol W}=\left(\boldsymbol{\Psi}^{\top} \boldsymbol{S}_u^{\top} \tilde{\boldsymbol{Q}} \boldsymbol{S}_u \boldsymbol{\Psi}+\boldsymbol{\Psi}^{\top} \boldsymbol R \boldsymbol \Psi\right)^{-1} \boldsymbol \Psi^{\top} \boldsymbol S_u^{\top} \tilde{\boldsymbol Q} \boldsymbol S_x, \quad \boldsymbol F=\boldsymbol\Psi \hat{\boldsymbol W},

by setting
:math:`\boldsymbol{u}=-\boldsymbol{F}\tilde{\boldsymbol{x}}_1`

Define related metrics
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def get_matrices(param: Dict, data: np.ndarray):
       # Task setting (tracking of acceleration profile and reaching of an end-point)
       Q = np.kron(np.identity(param["nbData"]),
                   np.diag(np.concatenate((np.zeros((param["nbVarU"] * 2)), np.ones(param["nbVarU"]) * 1e-6))))

       Q[-1 - param["nbVar"] + 1:-1 - param["nbVar"] + 2 * param["nbVarU"] + 1,
       -1 - param["nbVar"] + 1:-1 - param["nbVar"] + 2 * param["nbVarU"] + 1] = np.identity(2 * param["nbVarU"]) * 1e0

       # Weighting matrices in augmented state form
       Qm = np.zeros((param["nbVarX"] * param["nbData"], param["nbVarX"] * param["nbData"]))

       for t in range(param["nbData"]):
           id0 = np.linspace(0, param["nbVar"] - 1, param["nbVar"], dtype=int) + t * param["nbVar"]
           id = np.linspace(0, param["nbVarX"] - 1, param["nbVarX"], dtype=int) + t * param["nbVarX"]
           Qm[id[0]:id[-1] + 1, id[0]:id[-1] + 1] = np.vstack(
               (np.hstack((np.identity(param["nbVar"]), np.zeros((param["nbVar"], 1)))),
                np.append(-data[:, t].reshape(1, -1), 1))) \
                                                    @ block_diag((Q[id0[0]:id0[-1] + 1, id0[0]:id0[-1] + 1]),
                                                                 1) @ np.vstack(
               (np.hstack((np.identity(param["nbVar"]), -data[:, t].reshape(-1, 1))),
                np.append(np.zeros((1, param["nbVar"])), 1)))

       Rm = np.identity((param["nbData"] - 1) * param["nbVarU"]) * param["rfactor"]
       return Qm, Rm

.. _define-dynamical-system-2:

Define dynamical system
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def set_dynamical_system(param: Dict):
       A1d = np.zeros(param["nbDeriv"])
       for i in range(param["nbDeriv"]):
           A1d = A1d + np.diag(np.ones((1, param["nbDeriv"] - i)).flatten(), i) * param["dt"] ** i * 1 / factorial(
               i)  # Discrete 1D

       B1d = np.zeros((param["nbDeriv"], 1))
       for i in range(param["nbDeriv"]):
           B1d[param["nbDeriv"] - 1 - i] = param["dt"] ** (i + 1) * 1 / factorial(i + 1)  # Discrete 1D

       A0 = np.kron(A1d, np.eye(param["nbVarU"]))  # Discrete nD
       B0 = np.kron(B1d, np.eye(param["nbVarU"]))  # Discrete nD

       A = np.vstack((np.hstack((A0, np.zeros((param["nbVar"], 1)))),
                      np.hstack((np.zeros((param["nbVar"])), 1)).reshape(1, -1)))  # Augmented A (homogeneous)
       B = np.vstack((B0, np.zeros((1, param["nbVarU"]))))  # Augmented B (homogeneous)

       # Build Sx and Su transfer matrices (for augmented state space)
       Sx = np.kron(np.ones((param["nbData"], 1)), np.eye(param["nbVarX"], param["nbVarX"]))
       Su = np.zeros(
           (param["nbVarX"] * param["nbData"], param["nbVarU"] * (param["nbData"] - 1)))  # It's maybe n-1 not sure
       M = B
       for i in range(1, param["nbData"]):
           Sx[i * param["nbVarX"]:param["nbData"] * param["nbVarX"], :] = np.dot(
               Sx[i * param["nbVarX"]:param["nbData"] * param["nbVarX"], :], A)
           Su[param["nbVarX"] * i:param["nbVarX"] * i + M.shape[0], 0:M.shape[1]] = M
           M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]
       return Su, Sx, A, B

Solve LQT (CP version) in a feedback form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def get_u_x(param: Dict, state_noise: np.ndarray, muQ: np.ndarray, Qm: np.ndarray, Rm: np.ndarray, Su: np.ndarray,
               Sx: np.ndarray, PSI: np.ndarray, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
       # Least squares formulation of recursive LQR with an augmented state space and control primitives
       W = np.linalg.inv(PSI.T @ Su.T @ Qm @ Su @ PSI + PSI.T @ Rm @ PSI) @ PSI.T @ Su.T @ Qm @ Sx
       F = PSI @ W  # F with control primitives

       # Reproduction with feedback controller on augmented state space (with CP)
       Ka = np.empty((param["nbData"] - 1, param["nbVarU"], param["nbVarX"]))
       Ka[0, :, :] = F[0:param["nbVarU"], :]
       P = np.identity(param["nbVarX"])
       for t in range(param["nbData"] - 2):
           id = t * param["nbVarU"] + np.linspace(2, param["nbVarU"] + 1, param["nbVarU"], dtype=int)
           P = P @ np.linalg.pinv(A - B @ Ka[t, :, :])
           Ka[t + 1, :, :] = F[id, :] @ P

       x_hat = np.zeros((2, param["nbVarX"], param["nbData"] - 1))
       u_hat = np.zeros((2, param["nbVarPos"], param["nbData"] - 1))
       for n in range(2):
           x = np.append(muQ[:, 0] + np.append(np.array([2, 1]), np.zeros(param["nbVar"] - 2)), 1).reshape(-1, 1)
           for t in range(param["nbData"] - 1):
               # Feedback control on augmented state (resulting in feedback and feedforward terms on state)
               u = -Ka[t, :, :] @ x
               x = A @ x + B @ u  # Update of state vector
               if t == 24 and n == 1:
                   x = x + state_noise  # Simulated noise on the state
               x_hat[n, :, t] = x.flatten()  # State
       return u_hat, x_hat
