Linear Quadratic Tracking (LQT)
===============================

Linear quadratic tracking (LQT) is a simple form of optimal control that
trades off tracking and control costs expressed as quadratic terms over
a time horizon, with the evolution of the state described in a linear
form. The LQT problem is formulated as the minimization of the cost

.. math::


   \begin{aligned}
   c &=\left(\boldsymbol{\mu}_T-\boldsymbol{x}_T\right)^{\top} \boldsymbol{Q}_T\left(\boldsymbol{\mu}_T-\boldsymbol{x}_T\right)+\sum_{t=1}^{T-1}\left(\left(\boldsymbol{\mu}_t-\boldsymbol{x}_t\right)^{\top} \boldsymbol{Q}_t\left(\boldsymbol{\mu}_t-\boldsymbol{x}_t\right)+\boldsymbol{u}_t^{\top} \boldsymbol{R}_t \boldsymbol{u}_t\right) \\
   &=(\boldsymbol{\mu}-\boldsymbol{x})^{\top} \boldsymbol{Q}(\boldsymbol{\mu}-\boldsymbol{x})+\boldsymbol{u}^{\top} \boldsymbol{R} \boldsymbol{u},
   \end{aligned}

where
:math:`\boldsymbol{\mu}=\left[\boldsymbol{\mu}_{1}^{\top}, \boldsymbol{\mu}_{2}^{\top}, \ldots, \boldsymbol{\mu}_{T}^{\top}\right]^{\top}`
is the via-points planned by the pretrained DGform model and also the
targets need to be tracked,
:math:`\boldsymbol{x}=\left[\boldsymbol{x}_{1}^{\top}, \boldsymbol{x}_{2}^{\top}, \ldots, \boldsymbol{x}_{T}^{\top}\right]^{\top}`
refers to the evolution of the state variables,
:math:`\boldsymbol{u}=\left[\boldsymbol{u}_{1}^{\top}, \boldsymbol{u}_{2}^{\top}, \ldots, \boldsymbol{u}_{T-1}^{\top}\right]^{\top}`
is the evolution of control commands. :math:`\boldsymbol{Q}` represents
the precision matrices, and :math:`\boldsymbol{R}` is the control weight
matrices.

Define parameters
-----------------

.. code:: python

   param = {
       "nbData": 200,   # Number of data points
       "nbVarPos": 7,   # Dimension of position data
       "nbDeriv": 2,    # Number of static and dynamic features (2 -> [x,dx])
       "dt": 1e-2,      # Time step duration
       "rfactor": 1e-8  # Control cost
   }
   param["nb_var"] = param["nbVarPos"] * param["nbDeriv"]  # Dimension of state vector

Define related matrices
-----------------------

Control weight matrices :math:`\boldsymbol{R}`

.. code:: python

   R = np.identity((param["nbData"] - 1) * param["nbVarPos"], dtype=np.float32) * param[
   "rfactor"]

Task precision :math:`\boldsymbol{Q}`

.. code:: python

   Q = np.zeros((param["nb_var"] * param["nbData"], param["nb_var"] * param["nbData"]), dtype=np.float32)

Target
:math:`\boldsymbol{\mu}=\left[\boldsymbol{\mu}_{1}^{\top}, \boldsymbol{\mu}_{2}^{\top}, \ldots, \boldsymbol{\mu}_{T}^{\top}\right]^{\top}`

.. code:: python

   muQ = np.zeros((param["nb_var"] * param["nbData"], 1), dtype=np.float32)

   via_point = []
   for i in range(len(idx_slices)):
       slice_t = idx_slices[i]
       x_t = np.zeros((param["nb_var"], 1))
       x_t[:param["nbVarPos"]] = data[i].reshape((param["nbVarPos"], 1))
       muQ[slice_t] = x_t
       via_point.append(x_t)

       Q[slice_t, slice_t] = np.diag(np.hstack((np.ones(param["nbVarPos"]), np.zeros(param["nb_var"] - param["nbVarPos"]))))

Define linear system
--------------------

The simplified linear system can be described as
:math:`\boldsymbol{x}_{t+1}=\boldsymbol{A}_{t} \boldsymbol{x}_{t}+\boldsymbol{B}_{t} \boldsymbol{u}_{t}`.
By converting the time sequential evolution into matrix form, we can
yield a compact form,
:math:`\boldsymbol{x}=\boldsymbol{S}_{\boldsymbol{x}} \boldsymbol{x}_{1}+\boldsymbol{S}_{\boldsymbol{u}} u`.

In detail,

.. math::


   \begin{aligned}
   \boldsymbol{x}_2 &=\boldsymbol{A}_1 \boldsymbol{x}_1+\boldsymbol{B}_1 \boldsymbol{u}_1, \\
   \boldsymbol{x}_3 &=\boldsymbol{A}_2 \boldsymbol{x}_2+\boldsymbol{B}_2 \boldsymbol{u}_2=\boldsymbol{A}_2\left(\boldsymbol{A}_1 \boldsymbol{x}_1+\boldsymbol{B}_1 \boldsymbol{u}_1\right)+\boldsymbol{B}_2 \boldsymbol{u}_2 \\
   & \vdots \\
   \boldsymbol{x}_T &=\left(\prod_{t=1}^{T-1} \boldsymbol{A}_{T-t}\right) \boldsymbol{x}_1+\left(\prod_{t=1}^{T-2} \boldsymbol{A}_{T-t}\right) \boldsymbol{B}_1 \boldsymbol{u}_1+\left(\prod_{t=1}^{T-3} \boldsymbol{A}_{T-t}\right) \boldsymbol{B}_2 \boldsymbol{u}_2+\cdots+\boldsymbol{B}_{T-1} \boldsymbol{u}_{T-1},
   \end{aligned}

.. math::


   \underbrace{\left[\begin{array}{c}
   x_1 \\
   x_2 \\
   x_3 \\
   \vdots \\
   x_T
   \end{array}\right]}_{\boldsymbol{x}}=\underbrace{\left[\begin{array}{c}
   \boldsymbol{I} \\
   \boldsymbol{A}_1 \\
   \boldsymbol{A}_2 \boldsymbol{A}_1 \\
   \vdots \\
   \prod_{t=1}^{T-1} \boldsymbol{A}_{T-t}
   \end{array}\right]}_{S_{\boldsymbol{x}}} x_1+\underbrace{\left[\begin{array}{cccc}
   0 & 0 & \cdots & 0 \\
   \boldsymbol{B}_1 & 0 & \cdots & 0 \\
   \boldsymbol{A}_2 \boldsymbol{B}_1 & \boldsymbol{B}_2 & \cdots & 0 \\
   \vdots & \vdots & \ddots & \vdots \\
   \left(\prod_{t=1}^{T-2} \boldsymbol{A}_{T-t}\right) \boldsymbol{B}_1 & \left(\prod_{t=1}^{T-3} \boldsymbol{A}_{T-t}\right) \boldsymbol{B}_2 & \cdots & \boldsymbol{B}_{T-1}
   \end{array}\right]}_{S_u} \underbrace{\left[\begin{array}{c}
   u_1 \\
   u_2 \\
   \vdots \\
   u_{T-1}
   \end{array}\right]}_u,

.. code:: python

   def set_dynamical_system(param: Dict):
       A1d = np.zeros((param["nbDeriv"], param["nbDeriv"]), dtype=np.float32)
       B1d = np.zeros((param["nbDeriv"], 1), dtype=np.float32)
       for i in range(param["nbDeriv"]):
           A1d += np.diag(np.ones(param["nbDeriv"] - i), i) * param["dt"] ** i * 1 / factorial(i)
           B1d[param["nbDeriv"] - i - 1] = param["dt"] ** (i + 1) * 1 / factorial(i + 1)

       A = np.kron(A1d, np.identity(param["nbVarPos"], dtype=np.float32))
       B = np.kron(B1d, np.identity(param["nbVarPos"], dtype=np.float32))

       nb_var = param["nb_var"]  # Dimension of state vector

       # Build Sx and Su transfer matrices
       Su = np.zeros((nb_var * param["nbData"], param["nbVarPos"] * (param["nbData"] - 1)))
       Sx = np.kron(np.ones((param["nbData"], 1)), np.eye(nb_var, nb_var))

       M = B
       for i in range(1, param["nbData"]):
           Sx[i * nb_var:param["nbData"] * nb_var, :] = np.dot(Sx[i * nb_var:param["nbData"] * nb_var, :], A)
           Su[nb_var * i:nb_var * i + M.shape[0], 0:M.shape[1]] = M
           M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]

       return Su, Sx

Solve LQT
---------

Substituting
:math:`\boldsymbol{x}=\boldsymbol{S}_{\boldsymbol{x}} \boldsymbol{x}_{1}+\boldsymbol{S}_{\boldsymbol{u}} \boldsymbol{u}`
into Equ. 1, we can get the solution

.. math::


   \hat{\boldsymbol{u}}=\left(\boldsymbol{S}_{\boldsymbol{u}}^{\top} \boldsymbol{Q} \boldsymbol{S}_{\boldsymbol{u}}+\boldsymbol{R}\right)^{-1} \boldsymbol{S}_\boldsymbol{u}^{\top} \boldsymbol{Q}\left(\boldsymbol{\mu}-\boldsymbol{S}_{\boldsymbol{x}} \boldsymbol{x}_1\right)\\
   \hat{\boldsymbol{x}}=\boldsymbol{S}_{\boldsymbol{x}} \boldsymbol{x}_{1}+\boldsymbol{S}_{\boldsymbol{u}} \hat{\boldsymbol{u}}

The residuals of this least squares solution provides information about
the uncertainty of this estimate, in the form of a full covariance
matrix (at control trajectory level)

.. math::


   \hat{\boldsymbol{\Sigma}}^\boldsymbol{u}=\left(\boldsymbol{S}_\boldsymbol{u}^{\top} \boldsymbol{Q} \boldsymbol{S}_\boldsymbol{u}+\boldsymbol{R}\right)^{-1}

.. code:: python

   def get_u_x(param: Dict, start_pose: np.ndarray, muQ: np.ndarray, Q: np.ndarray, R: np.ndarray, Su: np.ndarray, Sx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
       x0 = start_pose.reshape((14, 1))

       u_hat = np.linalg.inv(Su.T @ Q @ Su + R) @ Su.T @ Q @ (muQ - Sx @ x0)
       # x= S_x x_1 + S_u u
       x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1, param["nb_var"]))
       return u_hat, x_hat
