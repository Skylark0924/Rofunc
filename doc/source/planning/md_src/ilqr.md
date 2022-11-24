# iterative Linear Quadratic Regulator (iLQR)

The LQT method mentioned in the previous blog only utilizes the optimization method as an interpolation method. However, the form of loss used by LQT is developed from Linear Quadratic Regulator (LQR) in optimal control theory. Except for the output interpolated trajectory, LQR provides the actual control commands while considering the robot's kinematics and dynamics. Compared with LQT, LQR has the advantage of directly constraining the control commands. The control commands can be velocities, acceleration and even jerks of robot joint angles or end-effectors. 

## iLQR formulation

Similar as the LQT, LQR defines and minimizes cost function $\sum^T_{t=1}c(\boldsymbol x_t, \boldsymbol u_t)$ for a dynamical system $\boldsymbol x_{t+1}=d(\boldsymbol x_t, \boldsymbol u_t)$, where $\boldsymbol x_t$ refers to the state and $\boldsymbol u_t$ refers to the control commands that drive the evolution of the state. The basic form of LQR is just designed for linear system $\dot{\boldsymbol x}=\boldsymbol A \boldsymbol x+ \boldsymbol B \boldsymbol u$. For nonlinear dynamical systems, iterative LQR is a solution by using *Taylor expansions* (or we can call it *Jacobian Linearization* in nonlinear control) [1]. 

Formally, we employ a first order Taylor expansion of the dynamical system around the point $(\hat {\boldsymbol x}_t, \hat{\boldsymbol u}_t)$:
$$
\begin{aligned}
\boldsymbol x_{t+1} & \approx d\left(\hat{\boldsymbol x}_t, \hat{\boldsymbol u}_t\right)+\frac{\partial d}{\partial \boldsymbol x_t}\left(\boldsymbol x_t-\hat{\boldsymbol x}_t\right)+\frac{\partial d}{\partial \boldsymbol{u}_t}\left(\boldsymbol{u}_t-\hat{\boldsymbol{u}}_t\right) \\
\Leftrightarrow \Delta \boldsymbol x_{t+1} & \approx \boldsymbol A_t \Delta \boldsymbol x_t+\boldsymbol B_t \Delta \boldsymbol u_t
\end{aligned}
$$
The cost function $c(\boldsymbol x_t, \boldsymbol u_t)$ can be approximated by a second order Taylor expansion around the point $(\hat {\boldsymbol x}_t, \hat{\boldsymbol u}_t)$:
$$
\begin{aligned}
&c\left(\boldsymbol x_t, \boldsymbol{u}_t\right) \approx c\left(\hat{\boldsymbol x}_t, \hat{\boldsymbol u}_t\right)+\Delta \boldsymbol x_t^{\mathrm{\prime}} \frac{\partial c}{\partial \boldsymbol x_t}+\Delta \boldsymbol{u}_t^{\mathrm{\prime}} \frac{\partial c}{\partial \boldsymbol{u}_t}+\frac{1}{2} \Delta \boldsymbol x_t^{\mathrm{\prime}} \frac{\partial^2 c}{\partial \boldsymbol x_t^2} \Delta \boldsymbol x_t+\Delta \boldsymbol{x}_t^{\mathrm{\prime}} \frac{\partial^2 c}{\partial \boldsymbol x_t \boldsymbol{u}_t} \Delta \boldsymbol{u}_t+\frac{1}{2} \Delta \boldsymbol{u}_t^{\mathrm{\prime}} \frac{\partial^2 c}{\partial \boldsymbol{u}_t^2} \Delta \boldsymbol{u}_t,\\
&\Leftrightarrow c\left(\boldsymbol x_t, \boldsymbol{u}_t\right) \approx c\left(\hat{\boldsymbol x}_t, \hat{\boldsymbol u}_t\right)+\frac{1}{2}\left[\begin{array}{c}
1 \\
\Delta \boldsymbol x_t \\
\Delta \boldsymbol u_t
\end{array}\right]^{\prime}\left[\begin{array}{ccc}
0 & \boldsymbol g_{\boldsymbol{x}, t}^{\top} & \boldsymbol g_{\boldsymbol{u}, t}^{\top} \\
\boldsymbol g_{\boldsymbol{x}, t} & \boldsymbol{H}_{\boldsymbol{x} \boldsymbol{x}, t} & \boldsymbol{H}_{\boldsymbol{ux}, t}^{\top} \\
\boldsymbol g_{\boldsymbol{u}, t} & \boldsymbol{H}_{\boldsymbol{ux},t} & \boldsymbol{H}_{\boldsymbol{uu}, t}
\end{array}\right]\left[\begin{array}{c}
1 \\
\Delta \boldsymbol x_t \\
\Delta \boldsymbol u_t
\end{array}\right],
\end{aligned}
$$
where $\boldsymbol{H}.$ are Hessian matrices, $\boldsymbol{g}.$ are gradients of the cost.

A solution in batch form can be computed by minimizing over $\boldsymbol u=[\boldsymbol u_1^\top, \boldsymbol u_2^\top, \dots, \boldsymbol u_{T-1}^\top]$, and represent the state as $\boldsymbol x=[\boldsymbol x_1^\top, \boldsymbol x_2^\top, \dots, \boldsymbol x_{T-1}^\top]$. The minimization problem can then be rewritten in batch form as
$$
\min _{\Delta \boldsymbol u} \Delta c(\Delta \boldsymbol x, \Delta \boldsymbol u), \quad \text { s.t. } \Delta \boldsymbol x=\boldsymbol S_{\boldsymbol u} \Delta \boldsymbol u
$$
By inserting the constraint into the cost, we obtain the optimization problem
$$
\min _{\Delta \boldsymbol u} \Delta \boldsymbol u^{\top} \boldsymbol S_{\boldsymbol u}^{\top} \boldsymbol g_{\boldsymbol x}+\Delta \boldsymbol u^{\top} \boldsymbol g_{\boldsymbol u}+\frac{1}{2} \Delta \boldsymbol u^{\top} \boldsymbol S_{\boldsymbol u}^{\top} \boldsymbol  H_{\boldsymbol x \boldsymbol x} \boldsymbol  S_{\boldsymbol u} \Delta \boldsymbol  u+\Delta \boldsymbol  u^{\top} \boldsymbol S_{\boldsymbol u}^{\top} \boldsymbol  H_{\boldsymbol u \boldsymbol x}^{\top} \Delta \boldsymbol  u+\frac{1}{2} \Delta \boldsymbol  u^{\top} \boldsymbol  H_{\boldsymbol u \boldsymbol u} \Delta \boldsymbol  u
$$
This optimization problem has the analytical solution:
$$
\boldsymbol S_{\boldsymbol u}^{\top} \boldsymbol g_{\boldsymbol x}+\boldsymbol g_{\boldsymbol u}+\frac{1}{2} \boldsymbol S_{\boldsymbol u}^{\top} \boldsymbol  H_{\boldsymbol x \boldsymbol x} \boldsymbol  S_{\boldsymbol u} \Delta \boldsymbol  u+ \boldsymbol S_{\boldsymbol u}^{\top} \boldsymbol  H_{\boldsymbol u \boldsymbol x}^{\top} \Delta \boldsymbol  u+\frac{1}{2}\boldsymbol  H_{\boldsymbol u \boldsymbol u} \Delta \boldsymbol  u=0\\
\Delta \hat {\boldsymbol u}= (\boldsymbol S_{\boldsymbol u}^{\top} \boldsymbol  H_{\boldsymbol x \boldsymbol x} \boldsymbol  S_{\boldsymbol u}+2\boldsymbol S_{\boldsymbol u}^{\top} \boldsymbol  H_{\boldsymbol u \boldsymbol x}^{\top}+\boldsymbol H_{\boldsymbol u \boldsymbol u})^{-1} (-\boldsymbol S_{\boldsymbol u}^{\top} \boldsymbol g_{\boldsymbol x}-\boldsymbol g_{\boldsymbol u})
$$
Thus, the training process of iLQR with batch formulation can be described as:

0. Initialize $\hat {\boldsymbol u}$, 
1. Compute $\hat {\boldsymbol x}$ by using $\hat {\boldsymbol u}, d(\cdot), \boldsymbol x_1$
2. Compute Jacobian Linearization $\boldsymbol A_t, \boldsymbol B_t$
3. Compute the compact matrix $\boldsymbol S_{\boldsymbol u}$ of $\boldsymbol A_t$ (similar as LQT)
4. Compute gradients $\boldsymbol g.$ and Hessian $\boldsymbol{H}.$
5. Compute $\Delta \hat {\boldsymbol u}$
6. Update $\hat {\boldsymbol u}$ with a learning rate $\alpha$: $\hat {\boldsymbol u}\leftarrow \hat {\boldsymbol u}+\alpha \Delta \hat {\boldsymbol u}$

7. Repeat 1~5 until $||\Delta \hat {\boldsymbol u}||<\Delta_\min$

## Implementation

In `rofunc`, we implement the iLQR for a single robot arm (or a robot with a specific end-effector). The main process is shown as follow:

```python
def uni(Mu, Rot, u0, x0, cfg, for_test=False):
    Q, R, idx, tl = get_matrices(cfg)
    Su0, Sx0 = set_dynamical_system(cfg)
    u, x = get_u_x(cfg, Mu, Rot, u0, x0, Q, R, Su0, Sx0, idx, tl)
    vis(cfg, Mu, Rot, x, tl, for_test=for_test)
```



```python
def get_u_x(cfg: DictConfig, Mu: np.ndarray, Rot: np.ndarray, u: np.ndarray, x0: np.ndarray, Q: np.ndarray,
            R: np.ndarray, Su0: np.ndarray, Sx0: np.ndarray, idx: np.ndarray, tl: np.ndarray):
    Su = Su0[idx.flatten()]  # We remove the lines that are out of interest

    for i in range(cfg.nbIter):
        x = Su0 @ u + Sx0 @ x0  # System evolution
        x = x.reshape([cfg.nbData, cfg.nbVarX])
        f, J = f_reach(cfg, x[tl], Mu, Rot)  # Residuals and Jacobians
        du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + R) @ (
                -Su.T @ J.T @ Q @ f.flatten() - u * cfg.rfactor)  # Gauss-Newton update
        # Estimate step size with backtracking line search method
        alpha = 2
        cost0 = f.flatten() @ Q @ f.flatten() + np.linalg.norm(u) ** 2 * cfg.rfactor  # Cost
        while True:
            utmp = u + du * alpha
            xtmp = Su0 @ utmp + Sx0 @ x0  # System evolution
            xtmp = xtmp.reshape([cfg.nbData, cfg.nbVarX])
            ftmp, _ = f_reach(cfg, xtmp[tl], Mu, Rot)  # Residuals
            cost = ftmp.flatten() @ Q @ ftmp.flatten() + np.linalg.norm(utmp) ** 2 * cfg.rfactor  # Cost
            if cost < cost0 or alpha < 1e-3:
                u = utmp
                print("Iteration {}, cost: {}".format(i, cost))
                break
            alpha /= 2
        if np.linalg.norm(du * alpha) < 1E-2:
            break  # Stop iLQR iterations when solution is reached
    return u, x
```



```python
def f_reach(cfg, robot_state, Mu, Rot, specific_robot=None):
    """
    Error and Jacobian for a via-points reaching task (in object coordinate system)
    Args:
        cfg:
        robot_state: joint state or Cartesian pose
    Returns:

    """
    if specific_robot is not None:
        ee_pose = specific_robot.fk(robot_state)
    else:
        ee_pose = fk(cfg, robot_state)
    f = logmap_2d(ee_pose, Mu)
    J = np.zeros([cfg.nbPoints * cfg.nbVarF, cfg.nbPoints * cfg.nbVarX])
    for t in range(cfg.nbPoints):
        f[t, :2] = Rot[t].T @ f[t, :2]  # Object-oriented forward kinematics
        Jtmp = Jacobian(cfg, robot_state[t])
        Jtmp[:2] = Rot[t].T @ Jtmp[:2]  # Object centered Jacobian

        if cfg.useBoundingBox:
            for i in range(2):
                if abs(f[t, i]) < cfg.sz[i]:
                    f[t, i] = 0
                    Jtmp[i] = 0
                else:
                    f[t, i] -= np.sign(f[t, i]) * cfg.sz[i]

        J[t * cfg.nbVarF:(t + 1) * cfg.nbVarF, t * cfg.nbVarX:(t + 1) * cfg.nbVarX] = Jtmp
    return f, J
```

### Reference

[1] Li, W., & Todorov, E. (2004, August). Iterative linear quadratic regulator design for nonlinear biological movement systems. In *ICINCO (1)* (pp. 222-229).

