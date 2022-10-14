import numpy as np
import rofunc as rf
import pbdlib as pbd
from typing import Union, List, Tuple


def get_dx(demos_x: Union[List, np.ndarray]):
    """
    Get the first derivative of the demo displacement
    Args:
        demos_x: demo displacement
    Returns:
        demos_dx: demo velocity
    """
    demos_dx = []
    for i in range(len(demos_x)):
        demo_dx = []
        for j in range(len(demos_x[i])):
            if 0 < j < len(demos_x[i]) - 1:
                dx = (demos_x[i][j + 1] - demos_x[i][j - 1]) / 2
            elif j == len(demos_x[i]) - 1:
                dx = demos_x[i][j] - demos_x[i][j - 1]
            else:
                dx = demos_x[i][j + 1] - demos_x[i][j]
            dx = dx / 0.01
            demo_dx.append(dx)
        demos_dx.append(np.array(demo_dx))
    return demos_dx


def get_A_b(demos_x: Union[List, np.ndarray]):
    demos_b = []
    demos_A = []
    for i in range(len(demos_x)):
        demos_b.append(np.tile(np.vstack([demos_x[i][0], demos_x[i][-1], ]), (len(demos_x[i]), 1, 1)))
        demos_A.append(np.tile([[[1., 0.], [-0., -1.]], [[1., 0.], [-0., -1.]]], (len(demos_x[i]), 1, 1, 1)))
    demos_A_xdx = [np.kron(np.eye(len(demos_x[0][0])), d) for d in demos_A]
    demos_b_xdx = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b]
    return demos_A, demos_b, demos_A_xdx, demos_b_xdx


def get_related_matrix(demos_x: Union[List, np.ndarray]):
    """
    Some related matrices are generated from the demo data with displacement
    M: Number of demonstrated trajectories in a training set (m will be used as index)
    T: Number of datapoints in a trajectory (t will be used as index)
    P: Number of candidate frames in a task-parameterized mixture (j will be used as index/exponent)
    nb_dim: Dimension of the demo state

    Args:
        demos_x: original demos with only displacement information, [M, T, nb_dim]
    Returns:
        demos_dx: first derivative of states, [M, T, nb_dim]
        demos_xdx: concat original states with their first derivative, [M, T, nb_dim * 2]
        demos_A: the orientation of the p-th candidate coordinate system for this demonstration, [M, T, P, 2, 2]
        demos_b: the position of the p-th candidate coordinate system for this demonstration,  [M, T, P, nb_dim]
        demos_A_xdx: augment demos_A to original states and their first derivative, [M, T, P, nb_dim * 2, nb_dim * 2]
        demos_b_xdx: augment demos_b to original states and their first derivative, [M, T, P, nb_dim * 2]
        demos_xdx_f: states and their first derivative in P frames, [M, T, P, nb_dim * 2]
        demos_xdx_augm: reshape demos_xdx_f, [M, T, nb_dim * 2 * P]
    """
    demos_dx = get_dx(demos_x)
    demos_xdx = [np.hstack([_x, _dx]) for _x, _dx in
                 zip(demos_x, demos_dx)]  # Position and velocity (num_of_points, 14)
    demos_A, demos_b, demos_A_xdx, demos_b_xdx = get_A_b(demos_x)
    demos_xdx_f = [np.einsum('taji,taj->tai', _A, _x[:, None] - _b) for _x, _A, _b in
                   zip(demos_xdx, demos_A_xdx, demos_b_xdx)]
    demos_xdx_augm = [d.reshape(-1, len(demos_xdx[0][0]) * 2) for d in
                      demos_xdx_f]  # (num_of_points, 28): 0~13 pos-vel in coord 1, 14~27 pos-vel in coord 2
    return demos_dx, demos_xdx, demos_A, demos_b, demos_A_xdx, demos_b_xdx, demos_xdx_f, demos_xdx_augm


def HMM_learning(demos_xdx_f: Union[List, np.ndarray], demos_xdx_augm: Union[List, np.ndarray], nb_states: int = 4,
                 reg: float = 1e-3, plot: bool = False) -> pbd.HMM:
    """
    Learn the HMM model by using demos_xdx_augm
    Args:
        demos_xdx_f: states and their first derivative in P frames, [M, T, P, nb_dim * 2]
        demos_xdx_augm: reshape demos_xdx_f, [M, T, nb_dim * 2 * P]
        nb_states: number of HMM states
        reg: [float] or list with [nb_dim x float] for different regularization in different dimensions
            Regularization term used in M-step for covariance matrices
        plot: [bool], whether to plot the demo and learned model
    Returns:
        model: learned HMM model
    """
    print('{}'.format('learning HMM model'))

    model = pbd.HMM(nb_states=nb_states)
    model.init_hmm_kbins(demos_xdx_augm)  # initializing model
    model.em(demos_xdx_augm, reg=reg)

    # plotting
    if plot:
        if int(len(demos_xdx_f[0][0, 0]) / 2) == 2:
            rf.tpgmm.hmm_plot(demos_xdx_f, model)
        elif int(len(demos_xdx_f[0][0, 0]) / 2) > 2:
            rf.tpgmm.hmm_plot_3d(demos_xdx_f, model, scale=0.1)
        else:
            raise Exception('Dimension is less than 2, cannot plot')
    return model


def poe(model: pbd.HMM, demos_A_xdx: Union[List, np.ndarray], demos_b_xdx: Union[List, np.ndarray],
        demos_x: Union[List, np.ndarray], show_demo_idx: int, plot: bool = False) -> pbd.GMM:
    """
    Product of Expert/Gaussian (PoE), which calculates the mixture distribution from multiple coordinates
    Args:
        model: learned model
        demos_A_xdx: augment demos_A to original states and their first derivative, [M, T, P, nb_dim * 2, nb_dim * 2]
        demos_b_xdx: augment demos_b to original states and their first derivative, [M, T, P, nb_dim * 2]
        demos_x: original demos with only displacement information, [M, T, nb_dim]
        demo_idx: index of the specific demo to be reproduced
        plot: [bool], whether to plot the PoE
    Returns:

    """
    print('{}'.format('product of expert'))

    # get transformation for given demonstration.
    # We use the transformation of the first timestep as they are constant
    A, b = demos_A_xdx[show_demo_idx][0], demos_b_xdx[show_demo_idx][0]
    # transformed model for coordinate system 1
    mod1 = model.marginal_model(slice(0, len(b[0]))).lintrans(A[0], b[0])
    # transformed model for coordinate system 2
    mod2 = model.marginal_model(slice(len(b[0]), len(b[0]) * 2)).lintrans(A[1], b[1])
    # product
    prod = mod1 * mod2

    if plot:
        if len(demos_x[0][0]) == 2:
            rf.tpgmm.poe_plot(mod1, mod2, prod, demos_x, show_demo_idx)
        elif len(demos_x[0][0]) > 2:
            rf.tpgmm.poe_plot_3d(mod1, mod2, prod, demos_x, show_demo_idx)
        else:
            raise Exception('Dimension is less than 2, cannot plot')
    return prod


def reproduce(model: pbd.HMM, prod: pbd.GMM, demos_x: Union[List, np.ndarray], demos_xdx: Union[List, np.ndarray],
              demos_xdx_augm: Union[List, np.ndarray], show_demo_idx: int, plot: bool = False) -> np.ndarray:
    """
    Reproduce the specific demo_idx from the learned model
    Args:
        model: learned model
        prod: result of PoE
        demos_x: original demos with only displacement information, [M, T, nb_dim]
        demos_xdx: concat original states with their first derivative, [M, T, nb_dim * 2]
        demos_xdx_augm: reshape demos_xdx_f, [M, T, nb_dim * 2 * P]
        show_demo_idx: index of the specific demo to be reproduced
        plot: [bool], whether to plot the
    Returns:
    """
    print('reproduce {}-th demo from learned representation'.format(show_demo_idx))

    # get the most probable sequence of state for this demonstration
    sq = model.viterbi(demos_xdx_augm[show_demo_idx])

    # solving LQR with Product of Gaussian, see notebook on LQR
    lqr = pbd.PoGLQR(nb_dim=len(demos_x[0][0]), dt=0.01, horizon=demos_xdx[show_demo_idx].shape[0])
    lqr.mvn_xi = prod.concatenate_gaussian(sq)  # augmented version of gaussian
    lqr.mvn_u = -4
    lqr.x0 = demos_xdx[show_demo_idx][0]

    xi = lqr.seq_xi
    if plot:
        if len(demos_x[0][0]) == 2:
            rf.tpgmm.generate_plot(xi, prod, demos_x, show_demo_idx)
        elif len(demos_x[0][0]) > 2:
            rf.tpgmm.generate_plot_3d(xi, prod, demos_x, show_demo_idx)
        else:
            raise Exception('Dimension is less than 2, cannot plot')
    return xi


def uni(demos_x: Union[List, np.ndarray], show_demo_idx: int, plot: bool = False) -> Tuple[pbd.HMM, np.ndarray]:
    """
    Learning the single arm/agent trajectory representation from demonstration via TP-GMM.
    Args:
        demos_x: multiple demonstration trajectories
        show_demo_idx: the index of demonstration to be reproduced
        plot: whether to plot

    Returns:

    """
    print('\033[1;32m--------{}--------\033[0m'.format(
        'Learning the trajectory representation from demonstration via TP-GMM'))

    _, demos_xdx, _, _, demos_A_xdx, demos_b_xdx, demos_xdx_f, demos_xdx_augm = get_related_matrix(demos_x)
    model = HMM_learning(demos_xdx_f, demos_xdx_augm, plot=plot)
    prod = poe(model, demos_A_xdx, demos_b_xdx, demos_x, show_demo_idx, plot=plot)
    rep = reproduce(model, prod, demos_x, demos_xdx, demos_xdx_augm, show_demo_idx, plot=plot)
    return model, rep


def bi(demos_left_x: Union[List, np.ndarray], demos_right_x: Union[List, np.ndarray], show_demo_idx: int,
       plot: bool = False) -> Tuple[pbd.HMM, pbd.HMM, np.ndarray, np.ndarray]:
    print('\033[1;32m--------{}--------\033[0m'.format(
        'Learning the bimanual trajectory representation from demonstration via TP-GMM'))

    _, demos_left_xdx, _, _, demos_left_A_xdx, demos_left_b_xdx, demos_left_xdx_f, demos_left_xdx_augm = get_related_matrix(
        demos_left_x)
    _, demos_right_xdx, _, _, demos_right_A_xdx, demos_right_b_xdx, demos_right_xdx_f, demos_right_xdx_augm = get_related_matrix(
        demos_right_x)

    model_l = HMM_learning(demos_left_xdx_f, demos_left_xdx_augm, plot=plot)
    model_r = HMM_learning(demos_right_xdx_f, demos_right_xdx_augm, plot=plot)

    prod_l = poe(model_l, demos_left_A_xdx, demos_left_b_xdx, demos_left_x, show_demo_idx, plot=plot)
    prod_r = poe(model_r, demos_right_A_xdx, demos_right_b_xdx, demos_right_x, show_demo_idx, plot=plot)

    rep_l = reproduce(model_l, prod_l, demos_left_x, demos_left_xdx, demos_left_xdx_augm, show_demo_idx, plot=plot)
    rep_r = reproduce(model_r, prod_r, demos_right_x, demos_right_xdx, demos_right_xdx_augm, show_demo_idx, plot=plot)

    if plot:
        nb_dim = int(rep_l.shape[1] / 2)
        data_lst = [rep_l[:, :nb_dim], rep_r[:, :nb_dim]]
        rf.visualab.traj_plot(data_lst)
    return model_l, model_r, rep_l, rep_r
