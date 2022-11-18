Usage
=====

Import the necessary packages:

.. code:: python

    import rofunc as rf
    import numpy as np
    from rofunc.config.utils import get_config

LQT variants
------------

LQT
~~~

.. code:: python

    via_points = ...  # [nb_via_points, via_points_dim]
    u_hat, x_hat, mu, idx_slices = rf.lqt.uni(via_points)
    rf.lqt.plot_3d_uni(x_hat, mu, idx_slices, ori=False, save=False)

LQT with feedback
~~~~~~~~~~~~~~~~~

.. code:: python

    via_points = ...  # [nb_via_points, via_points_dim]
    rf.lqt.uni_fb(via_points)

LQT with control primitive 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    via_points = ...  # [nb_via_points, via_points_dim]
    u_hat, x_hat, mu, idx_slices = rf.lqt.uni_cp(via_points)
    rf.lqt.plot_3d_uni([x_hat], mu, idx_slices)

iLQR variants
-------------

iLQR
~~~~

.. code:: python

    cfg = get_config('./planning', 'ilqr')
    # via-points
    Mu = np.array([[2, 1, -np.pi / 6], [3, 2, -np.pi / 3]])  # Via-points
    Rot = np.zeros([cfg.nbPoints, 2, 2])  # Object orientation matrices
    # Object rotation matrices
    for t in range(cfg.nbPoints):
        orn_t = Mu[t, -1]
        Rot[t, :, :] = np.asarray([
            [np.cos(orn_t), -np.sin(orn_t)],
            [np.sin(orn_t), np.cos(orn_t)]
        ])
    u0 = np.zeros(cfg.nbVarU * (cfg.nbData - 1))  # Initial control command
    x0 = np.array([3 * np.pi / 4, -np.pi / 2, -np.pi / 4])  # Initial state
    rf.lqr.uni(Mu, Rot, u0, x0, cfg)

iLQR with control primitive 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    cfg = get_config('./planning', 'ilqr')

    # Via-points
    Mu = np.array([[2, 1, -np.pi / 2], [3, 1, -np.pi / 2]])  # Via-points
    Rot = np.zeros([2, 2, cfg.nbPoints])  # Object orientation matrices

    # Object rotation matrices
    for t in range(cfg.nbPoints):
        orn_t = Mu[t, -1]
        Rot[t] = np.asarray([
            [np.cos(orn_t), -np.sin(orn_t)],
            [np.sin(orn_t), np.cos(orn_t)]
        ])
    u0 = np.zeros(cfg.nbVarU * (cfg.nbData - 1))  # Initial control command
    x0 = np.array([3 * np.pi / 4, -np.pi / 2, -np.pi / 4])  # Initial state
    rf.lqr.uni_cp(Mu, Rot, u0, x0, cfg)

MPC variants
------------
