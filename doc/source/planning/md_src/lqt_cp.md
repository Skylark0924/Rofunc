# LQT (control primitive version)

Use the control primitive to represent the control command $\boldsymbol{u}=\Psi \boldsymbol{w}$ 
$$
\min _{\boldsymbol{w}}(\boldsymbol{x}-\boldsymbol{\mu})^{\top} \boldsymbol{Q}(\boldsymbol{x}-\boldsymbol{\mu})+\boldsymbol{w}^{\top} \Psi^{\top} \boldsymbol{R} \Psi \boldsymbol{w}, \quad \text { s.t. } \quad \boldsymbol{x}=\boldsymbol{S}_{\boldsymbol{x}} \boldsymbol{x}_1+\boldsymbol{S}_{\boldsymbol{u}} \Psi \boldsymbol{w}
$$
the analytic solution is 
$$
\hat{w}=\left(\Psi^{\top} \boldsymbol{S}_u^{\top} Q \boldsymbol{S}_u \Psi+\Psi^{\top} \boldsymbol{R} \Psi\right)^{-1} \Psi^{\top} \boldsymbol{S}_u^{\top} Q\left(\mu-S_x x_1\right)
$$

## Define control primitives

```python
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
```

## Define dynamical system

```python
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
```

## Solve the LQT with control primitive 

$$
\hat{w}=\left(\Psi^{\top} \boldsymbol{S}_u^{\top} Q \boldsymbol{S}_u \Psi+\Psi^{\top} \boldsymbol{R} \Psi\right)^{-1} \Psi^{\top} \boldsymbol{S}_u^{\top} Q\left(\mu-S_x x_1\right)
$$

```python
x0 = start_pose.reshape((14, 1))
w_hat = np.linalg.inv(PSI.T @ Su.T @ Q @ Su @ PSI + PSI.T @ R @ PSI) @ PSI.T @ Su.T @ Q @ (muQ - Sx @ x0)
u_hat = PSI @ w_hat
x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1, param["nb_var"]))
u_hat = u_hat.reshape((-1, param["nbVarPos"]))
```

