# LQT (feedback version)


$$
\begin{aligned}
c &=\left(\boldsymbol{\mu}_T-\boldsymbol{x}_T\right)^{\top} \boldsymbol{Q}_T\left(\boldsymbol{\mu}_T-\boldsymbol{x}_T\right)+\sum_{t=1}^{T-1}\left(\left(\boldsymbol{\mu}_t-\boldsymbol{x}_t\right)^{\top} \boldsymbol{Q}_t\left(\boldsymbol{\mu}_t-\boldsymbol{x}_t\right)+\boldsymbol{u}_t^{\top} \boldsymbol{R}_t \boldsymbol{u}_t\right) \\
&=\tilde{\boldsymbol{x}}_T^{\top} \tilde{\boldsymbol{Q}}_T \tilde{\boldsymbol{x}}_T+\sum_{t=1}^{T-1}\left(\tilde{\boldsymbol{x}}_t^{\top} \tilde{\boldsymbol{Q}}_t \tilde{\boldsymbol{x}}_t+\boldsymbol{u}_t^{\top} \boldsymbol{R}_t \boldsymbol{u}_t\right),
\end{aligned}
$$

## Define related matrices

Control weight matrices $\boldsymbol{R}$ 

```python
R = np.eye(param["nbVarPos"]) * param["rfactor"]  # Control cost matrix
```

Task precision $\boldsymbol{Q}$ 
$$
\tilde{\boldsymbol{Q}}_t=\left[\begin{array}{cc}
\boldsymbol{Q}_t^{-1}+\boldsymbol{\mu}_t \boldsymbol{\mu}_t^{\top} & \boldsymbol{\mu}_t \\
\boldsymbol{\mu}_t^{\top} & 1
\end{array}\right]^{-1}=\left[\begin{array}{cc}
\boldsymbol{I} & 0 \\
-\boldsymbol{\mu}_t^{\top} & 1
\end{array}\right]\left[\begin{array}{cc}
\boldsymbol{Q}_t & 0 \\
0 & 1
\end{array}\right]\left[\begin{array}{cc}
\boldsymbol{I} & -\boldsymbol{\mu}_t \\
0 & 1
\end{array}\right],
$$

```python
# Definition of augmented precision matrix Qa based on standard precision matrix Q0
Q0 = np.diag(np.hstack([np.ones(param["nbVarPos"]), np.zeros(param["nbVar"] - param["nbVarPos"])]))

Q0_augmented = np.identity(param["nbVar"] + 1)
Q0_augmented[:param["nbVar"], :param["nbVar"]] = Q0

Q = np.zeros([param["nbVar"] + 1, param["nbVar"] + 1, param["nbData"]])
for i in range(param["nbPoints"]):
    Q[:, :, int(tl[i])] = np.vstack([
        np.hstack([np.identity(param["nbVar"]), np.zeros([param["nbVar"], 1])]),
        np.hstack([-Mu[i, :], 1])]) @ Q0_augmented @ np.vstack([
        np.hstack([np.identity(param["nbVar"]), -Mu[i, :].reshape([-1, 1])]),
        np.hstack([np.zeros(param["nbVar"]), 1])])
```

## Define dynamical system

$$
\underbrace{\left[\begin{array}{c}
\boldsymbol{x}_{t+1} \\
1
\end{array}\right]}_{\tilde{\boldsymbol{x}}_{t+1}}=\underbrace{\left[\begin{array}{cc}
\boldsymbol{A} & \mathbf{0} \\
\mathbf{0} & 1
\end{array}\right]}_{\tilde{\boldsymbol{A}}} \underbrace{\left[\begin{array}{c}
\boldsymbol{x}_t \\
1
\end{array}\right]}_{\tilde{\boldsymbol{x}}_t}+\underbrace{\left[\begin{array}{c}
\boldsymbol{B} \\
0
\end{array}\right]}_{\tilde{\boldsymbol{B}}} \boldsymbol{u}_t
$$

```python
def set_dynamical_system(param: Dict):
    A1d = np.zeros((param["nbDeriv"], param["nbDeriv"]))
    for i in range(param["nbDeriv"]):
        A1d += np.diag(np.ones((param["nbDeriv"] - i,)), i) * param["dt"] ** i / np.math.factorial(i)  # Discrete 1D

    B1d = np.zeros((param["nbDeriv"], 1))
    for i in range(0, param["nbDeriv"]):
        B1d[param["nbDeriv"] - i - 1, :] = param["dt"] ** (i + 1) * 1 / np.math.factorial(i + 1)  # Discrete 1D

    A0 = np.kron(A1d, np.eye(param["nbVarPos"]))  # Discrete nD
    B0 = np.kron(B1d, np.eye(param["nbVarPos"]))  # Discrete nD
    A = np.eye(A0.shape[0] + 1)  # Augmented A
    A[:A0.shape[0], :A0.shape[1]] = A0
    B = np.vstack((B0, np.zeros((1, param["nbVarPos"]))))  # Augmented B
    return A, B
```

## Solve LQT with feedback gain


$$
\hat{v}_t=\min _{\boldsymbol{u}_t}\left[\begin{array}{l}
\boldsymbol{x}_t \\
\boldsymbol{u}_t
\end{array}\right]^{\top}\left[\begin{array}{ll}
\boldsymbol{Q}_{\boldsymbol{x} \boldsymbol{x}, t} & Q_{u \boldsymbol{x}, t}^{\top} \\
\boldsymbol{Q}_{\boldsymbol{u x}, t} & Q_{u \boldsymbol{u}, t}
\end{array}\right]\left[\begin{array}{l}
\boldsymbol{x}_t \\
\boldsymbol{u}_t
\end{array}\right], \quad \text { where } \quad\left\{\begin{array}{l}
Q_{\boldsymbol{x} \boldsymbol{x}, t}=\boldsymbol{A}_t^{\top} \boldsymbol{V}_{t+1} \boldsymbol{A}_t+\boldsymbol{Q}_t, \\
Q_{u \boldsymbol{u}, t}=\boldsymbol{B}_t^{\top} \boldsymbol{V}_{t+1} \boldsymbol{B}_t+\boldsymbol{R}_t, \\
Q_{\boldsymbol{u x}, t}=\boldsymbol{B}_t^{\top} \boldsymbol{V}_{t+1} \boldsymbol{A}_t
\end{array}\right.
$$

$$
\hat{u}_t=-\boldsymbol{K}_t x_t, \quad \text { with } \quad \boldsymbol{K}_t=Q_{u \boldsymbol{u}, t}^{-1} Q_{\boldsymbol{u x}, t} .
$$

```python
P = np.zeros((param["nbVarX"], param["nbVarX"], param["nbData"]))
P[:, :, -1] = Q[:, :, -1]

for t in range(param["nbData"] - 2, -1, -1):
	P[:, :, t] = Q[:, :, t] - A.T @ (
            P[:, :, t + 1] @ np.dot(B, np.linalg.pinv(B.T @ P[:, :, t + 1] @ B + R))
            @ B.T @ P[:, :, t + 1] - P[:, :, t + 1]) @ A
```


$$
\hat{\boldsymbol{u}}_t=-\tilde{\boldsymbol{K}}_t \tilde{\boldsymbol{x}}_t=\boldsymbol{K}_t\left(\boldsymbol{\mu}_t-\boldsymbol{x}_t\right)+\boldsymbol{u}_t^{\mathrm{ff}}
$$

$$
\boldsymbol{u}_t^{\mathrm{ff}}=-\boldsymbol{k}_t-\boldsymbol{K}_t \boldsymbol{\mu}_t
$$

```python
def get_u_x(param: Dict, state_noise: np.ndarray, P: np.ndarray, R: np.ndarray, A: np.ndarray, B: np.ndarray):
    x_hat = np.zeros((param["nbVar"] + 1, 2, param["nbData"]))
    u_hat = np.zeros((param["nbVarPos"], 2, param["nbData"]))
    for n in range(2):
        x = np.hstack([np.zeros(param["nbVar"]), 1])
        for t in range(param["nbData"]):
            Z_bar = B.T @ P[:, :, t] @ B + R
            K = np.linalg.inv(Z_bar.T @ Z_bar) @ Z_bar.T @ B.T @ P[:, :, t] @ A  # Feedback gain
            u = -K @ x  # Acceleration command with FB on augmented state (resulting in feedback and feedforward terms)
            x = A @ x + B @ u  # Update of state vector

            if t == 25 and n == 1:
                x += state_noise

            x_hat[:, n, t] = x  # Log data
            u_hat[:, n, t] = u  # Log data
    return u_hat, x_hat
```

