import numpy as np
import rofunc as rf
from scipy.linalg import block_diag

import pbdlib as pbd


def poe_gmr(mu_gmr, sigma_gmr, start_x, end_x, demos_x, demo_idx, horzion=150, plot=False):
    gmm = pbd.GMM(mu=mu_gmr, sigma=sigma_gmr)

    # get transformation for new situation
    b = np.tile(np.vstack([start_x, end_x, ]), (horzion, 1, 1))
    A = np.tile([[[1., 0.], [-0., -1.]], [[1., 0.], [-0., -1.]]], (horzion, 1, 1, 1))

    b_xdx = np.concatenate([b, np.zeros(b.shape)], axis=-1)
    A_xdx = np.kron(np.eye(len(demos_x[0][0])), A)

    A, b = A_xdx[0], b_xdx[0]

    # transformed model for coordinate system 1
    mod1 = gmm.marginal_model(slice(0, len(b[0]))).lintrans(A[0], b[0])
    # transformed model for coordinate system 2
    mod2 = gmm.marginal_model(slice(len(b[0]), len(b[0]) * 2)).lintrans(A[1], b[1])
    # product
    prod = mod1 * mod2

    if plot:
        if len(demos_x[0][0]) == 2:
            rf.tpgmm.poe_plot(mod1, mod2, prod, demos_x, demo_idx)
        elif len(demos_x[0][0]) > 2:
            rf.tpgmm.poe_plot_3d(mod1, mod2, prod, demos_x, demo_idx)
        else:
            raise Exception('Dimension is less than 2, cannot plot')
    return prod


def uni(demos_x, show_demo_idx, start_x, end_x, plot=False):
    # Learn the time-dependent GMR from demonstration
    demos_dx, demos_xdx, demos_A, demos_b, demos_A_xdx, demos_b_xdx, demos_xdx_f, demos_xdx_augm = rf.tpgmm.get_related_matrix(
        demos_x)
    # demos_xdx_augm, demos_dx, demos_xdx, t = rf.gmr.traj_align(demos_xdx_augm, demos_dx, demos_xdx, plot=plot)
    t = np.linspace(0, 50, demos_x[0].shape[0])
    demos = [np.hstack([t[:, None], d]) for d in demos_xdx_augm]

    model = rf.gmr.GMM_learning(demos, plot=plot)
    mu_gmr, sigma_gmr = rf.gmr.estimate(model, demos_xdx_f, t[:, None], dim_in=slice(0, 1),
                                        dim_out=slice(1, 4 * len(start_x) + 1), plot=plot)

    # Generate for new situation
    prod = poe_gmr(mu_gmr, sigma_gmr, start_x, end_x, demos_x, show_demo_idx,
                   horzion=demos_xdx[show_demo_idx].shape[0], plot=plot)

    lqr = pbd.PoGLQR(nb_dim=len(demos_x[0][0]), dt=0.01, horizon=demos_xdx[show_demo_idx].shape[0])

    mvn = pbd.MVN()
    mvn.mu = np.concatenate([i for i in prod.mu])
    mvn._sigma = block_diag(*[i for i in prod.sigma])

    lqr.mvn_xi = mvn
    lqr.mvn_u = -4
    start_xdx = np.zeros(len(start_x) * 2)
    start_xdx[:len(start_x)] = start_x
    lqr.x0 = start_xdx

    xi = lqr.seq_xi
    # end_xdx = np.zeros(len(end_x) * 2)
    # end_xdx[:len(start_x)] = end_x
    # xi = np.append(xi, end_xdx.reshape((1, -1)), axis=0)
    if plot:
        if len(demos_x[0][0]) == 2:
            rf.tpgmm.generate_plot(xi, prod, demos_x, show_demo_idx)
        elif len(demos_x[0][0]) > 2:
            rf.tpgmm.generate_plot_3d(xi, prod, demos_x, show_demo_idx)
        else:
            raise Exception('Dimension is less than 2, cannot plot')
    return prod, xi


if __name__ == '__main__':
    # Uni
    # demo_points = np.array([[[0, 0], [-1, 8], [4, 3], [2, 1], [4, 3]],
    #                         [[0, -2], [-1, 7], [3, 2.5], [2, 1.6], [4, 3]],
    #                         [[0, -1], [-1, 8], [4, 5.2], [2, 1.1], [4, 3.5]]])
    # demos_x = rf.data_generator.multi_bezier_demos(demo_points)  # (3, 50, 2): 3 demos, each has 50 points
    # start_pose, end_pose = [-1, -2], [6, 6]
    # model, rep = uni(demos_x, 2, start_pose, end_pose, plot=True)

    # Uni_3d
    raw_demo = np.load('/home/ubuntu/Data/2022_09_09_Taichi/xsens_mvnx/010-058/LeftHand.npy')
    raw_demo = np.expand_dims(raw_demo, axis=0)
    demos_x = np.vstack((raw_demo[:, 82:232, :], raw_demo[:, 233:383, :], raw_demo[:, 376:526, :]))
    show_demo_idx = 2
    start_pose = demos_x[show_demo_idx][-1]
    end_pose = demos_x[show_demo_idx][-1]
    model, rep = uni(demos_x, show_demo_idx, start_pose, end_pose, plot=True)
