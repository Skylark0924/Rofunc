import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from rofunc.utils.bezier import multi_bezier_demos

import pbdlib as pbd


# demo_idx = 2


def get_rel_demos(left_x, right_x):
    # calculate the relative movement of each demo
    rel_demos = np.zeros_like(left_x)
    plt.figure()
    for i in range(left_x.shape[0]):
        for j in range(left_x.shape[1]):
            # rel_demos[i, j] = np.linalg.norm(left_x[i, j] - right_x[i, j])
            rel_demos[i, j] = left_x[i, j] - right_x[i, 49 - j]

        plt.plot(rel_demos[i, :, 0], rel_demos[i, :, 1])
    plt.legend()
    plt.show()
    return rel_demos


def get_dx(demos_x):
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


def get_A_b(demos_x):
    demos_b = []
    demos_A = []
    for i in range(len(demos_x)):
        demos_b.append(np.tile(np.vstack([demos_x[i][0], demos_x[i][-1], ]), (len(demos_x[i]), 1, 1)))
        demos_A.append(np.tile([[[1., 0.], [-0., -1.]], [[1., 0.], [-0., -1.]]], (len(demos_x[i]), 1, 1, 1)))
    demos_A_xdx = [np.kron(np.eye(2), d) for d in demos_A]
    demos_b_xdx = [np.concatenate([d, np.zeros(d.shape)], axis=-1) for d in demos_b]

    return demos_A, demos_b, demos_A_xdx, demos_b_xdx


def get_related_matrix(demos_x):
    demos_dx = get_dx(demos_x)
    demos_xdx = [np.hstack([_x, _dx]) for _x, _dx in zip(demos_x, demos_dx)]  # Position and velocity (num_of_points, 4)
    demos_A, demos_b, demos_A_xdx, demos_b_xdx = get_A_b(demos_x)
    demos_xdx_f = [np.einsum('taji,taj->tai', _A, _x[:, None] - _b) for _x, _A, _b in
                   zip(demos_xdx, demos_A_xdx, demos_b_xdx)]
    demos_xdx_augm = [d.reshape(-1, 8) for d in
                      demos_xdx_f]  # (num_of_points, 8): 0~3 pos-vel in coord 1, 4~7 pos-vel in coord 2
    return demos_xdx, demos_A_xdx, demos_b_xdx, demos_xdx_f, demos_xdx_augm


def HMM_learning(demos_xdx_f, demos_xdx_augm, plot=False):
    model = pbd.HMM(nb_states=4, nb_dim=8)
    model.init_hmm_kbins(demos_xdx_augm)  # initializing model

    model.em(demos_xdx_augm, reg=1e-3)

    # plotting
    if plot:
        fig, ax = plt.subplots(ncols=2, nrows=1)
        fig.set_size_inches(12, 6)

        # position plotting
        ax[0].set_title('pos - coord. %d' % 1)
        for p in demos_xdx_f:
            ax[0].plot(p[:, 0, 0], p[:, 0, 1])
        pbd.plot_gmm(model.mu, model.sigma, ax=ax[0], dim=[0, 1], color='steelblue')

        ax[1].set_title('pos - coord. %d' % 2)
        for p in demos_xdx_f:
            ax[1].plot(p[:, 1, 0], p[:, 1, 1])
        pbd.plot_gmm(model.mu, model.sigma, ax=ax[1], dim=[4, 5], color='orangered')

        plt.tight_layout()
        plt.show()
    return model


def poe(model, demos_A_xdx, demos_b_xdx, demos_x, demo_idx, plot=False):
    # global demo_idx
    # get transformation for given demonstration.
    # We use the transformation of the first timestep as they are constant
    A, b = demos_A_xdx[demo_idx][0], demos_b_xdx[demo_idx][0]
    # transformed model for coordinate system 1
    mod1 = model.marginal_model(slice(0, 4)).lintrans(A[0], b[0])
    # transformed model for coordinate system 2
    mod2 = model.marginal_model(slice(4, 8)).lintrans(A[1], b[1])
    # product
    prod = mod1 * mod2

    fig, ax = plt.subplots(ncols=3, nrows=1)
    fig.set_size_inches((12, 6))
    for i in ax:
        # for j in i:
        i.set_aspect('equal')

    # omega = np.array([1 / 8, 1 / 8, 1 / 8, 1 / 8])
    # sigma_new = []
    # for i in range(len(model.sigma)):
    #     sigma_new.append(omega * model.sigma[i, :4, :4].reshape((4, 4)))

    # sigma_new = np.array(sigma_new)

    if plot:
        ax[0].set_title('model 1')
        pbd.plot_gmm(model.mu, model.sigma, swap=True, ax=ax[0], dim=[0, 1], color='steelblue', alpha=0.3)
        ax[1].set_title('model 2')
        pbd.plot_gmm(model.mu, model.sigma, swap=True, ax=ax[1], dim=[4, 5], color='orangered', alpha=0.3)
        # ax[1, 0].set_title('model 1 after scaling')
        # pbd.plot_gmm(model.mu, sigma_new, swap=True, ax=ax[1, 0], dim=[0, 1], color='blue', alpha=0.3)
        # ax[1, 1].set_title('model 2')
        # pbd.plot_gmm(model.mu, model.sigma, swap=True, ax=ax[1, 1], dim=[4, 5], color='orangered', alpha=0.3)

        ax[2].set_title('tranformed models and product')
        # sigma_mod1_new = []
        # for i in range(len(mod1.sigma)):
        #     sigma_mod1_new.append(omega * mod1.sigma[i, :4, :4].reshape((4, 4)))
        # product

        # mod1_new = model.marginal_model(slice(0, 4)).lintrans(A[0], b[0])
        # mod1_new.sigma = np.array(sigma_mod1_new)
        # new_prod = mod1_new * mod2
        # print(prod.sigma)
        # prod = mod1 * mod2

        pbd.plot_gmm(mod1.mu, mod1.sigma, swap=True, ax=ax[2], dim=[0, 1], color='steelblue', alpha=0.3)
        pbd.plot_gmm(mod2.mu, mod2.sigma, swap=True, ax=ax[2], dim=[0, 1], color='orangered', alpha=0.3)
        pbd.plot_gmm(prod.mu, prod.sigma, swap=True, ax=ax[2], dim=[0, 1], color='gold', alpha=0.3)
        ax[2].plot(demos_x[demo_idx][:, 0], demos_x[demo_idx][:, 1], color="b")

        # ax[1, 2].set_title('tranformed models and product')
        # pbd.plot_gmm(mod1_new.mu, mod1_new.sigma, swap=True, ax=ax[1, 2], dim=[0, 1], color='blue', alpha=0.3)
        # pbd.plot_gmm(new_prod.mu, new_prod.sigma, swap=True, ax=ax[1, 2], dim=[0, 1], color='green', alpha=0.3)
        # pbd.plot_gmm(mod2.mu, mod2.sigma, swap=True, ax=ax[1, 2], dim=[0, 1], color='orangered', alpha=0.3)
        # ax[1, 2].plot(demos_x[4][:, 0], demos_x[4][:, 1], color="b")

        patches = [mpatches.Patch(color='steelblue', label='transformed model 1'),
                   mpatches.Patch(color='orangered', label='transformed model 2'),
                   mpatches.Patch(color='gold', label='product')]

        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
    return prod


def generate(model, prod, demos_x, demos_xdx, demos_xdx_augm, demo_idx, plot=False):
    # get the most probable sequence of state for this demonstration
    sq = model.viterbi(demos_xdx_augm[demo_idx])

    # solving LQR with Product of Gaussian, see notebook on LQR
    lqr = pbd.PoGLQR(nb_dim=2, dt=0.01, horizon=demos_xdx[demo_idx].shape[0])
    lqr.mvn_xi = prod.concatenate_gaussian(sq)  # augmented version of gaussian
    lqr.mvn_u = -4
    lqr.x0 = demos_xdx[demo_idx][0]

    xi = lqr.seq_xi
    if plot:
        plt.figure()

        plt.title('Trajectory reproduction')
        plt.plot(xi[:, 0], xi[:, 1], color='r', lw=2, label='generated line')
        # pbd.plot_gmm(mod1.mu, mod1.sigma, swap=True, ax=ax[0], dim=[0, 1], color='steelblue', alpha=0.3)
        # pbd.plot_gmm(mod2.mu, mod2.sigma, swap=True, ax=ax[0], dim=[0, 1], color='orangered', alpha=0.3)
        pbd.plot_gmm(prod.mu, prod.sigma, swap=True, dim=[0, 1], color='gold')
        plt.plot(demos_x[demo_idx][:, 0], demos_x[demo_idx][:, 1], 'k--', lw=2, label='demo line')
        plt.axis('equal')
        plt.legend()
        plt.show()
    return xi


def uni(demos_x, show_demo_idx, plot=False):
    demos_xdx, demos_A_xdx, demos_b_xdx, demos_xdx_f, demos_xdx_augm = get_related_matrix(demos_x)
    model = HMM_learning(demos_xdx_f, demos_xdx_augm, plot=plot)
    prod = poe(model, demos_A_xdx, demos_b_xdx, demos_x, show_demo_idx, plot=plot)
    gen = generate(model, prod, demos_x, demos_xdx, demos_xdx_augm, show_demo_idx, plot=plot)

    if plot:
        plt.figure()
        plt.plot(gen[:, 0], gen[:, 1])
        plt.show()
    return gen


def bi(demos_left_x, demos_right_x, show_demo_idx, plot=False):
    demos_left_xdx, demos_left_A_xdx, demos_left_b_xdx, demos_left_xdx_f, demos_left_xdx_augm = get_related_matrix(
        demos_left_x)
    demos_right_xdx, demos_right_A_xdx, demos_right_b_xdx, demos_right_xdx_f, demos_right_xdx_augm = get_related_matrix(
        demos_right_x)

    model_l = HMM_learning(demos_left_xdx_f, demos_left_xdx_augm, plot=plot)
    model_r = HMM_learning(demos_right_xdx_f, demos_right_xdx_augm, plot=plot)

    prod_l = poe(model_l, demos_left_A_xdx, demos_left_b_xdx, demos_left_x, show_demo_idx, plot=plot)
    prod_r = poe(model_r, demos_right_A_xdx, demos_right_b_xdx, demos_right_x, show_demo_idx, plot=plot)

    gen_l = generate(model_l, prod_l, demos_left_x, demos_left_xdx, demos_left_xdx_augm, show_demo_idx, plot=plot)
    gen_r = generate(model_r, prod_r, demos_right_x, demos_right_xdx, demos_right_xdx_augm, show_demo_idx, plot=plot)

    if plot:
        plt.figure()
        plt.plot(gen_l[:, 0], gen_l[:, 1])
        plt.plot(gen_r[:, 0], gen_r[:, 1])
        plt.show()
    return gen_l, gen_r


if __name__ == '__main__':
    import rofunc as rf
    # Uni
    # demo_points = np.array([[[0, 0], [-1, 8], [4, 3], [2, 1], [4, 3]],
    #                         [[0, -2], [-1, 7], [3, 2.5], [2, 1.6], [4, 3]],
    #                         [[0, -1], [-1, 8], [4, 5.2], [2, 1.1], [4, 3.5]]])
    # demos_x = rf.utils.bezier.multi_bezier_demos(demo_points)  # (3, 50, 2): 3 demos, each has 50 points
    # rf.tpgmm.uni(demos_x, show_demo_idx=2, plot=True)

    # Bi
    # left_demo_points = np.array([[[0, 0], [-1, 8], [4, 3], [2, 1], [4, 3]],
    #                              [[0, -2], [-1, 7], [3, 2.5], [2, 1.6], [4, 3]],
    #                              [[0, -1], [-1, 8], [4, 5.2], [2, 1.1], [4, 3.5]]])
    # right_demo_points = np.array([[[8, 8], [7, 1], [4, 3], [6, 8], [4, 3]],
    #                               [[8, 7], [7, 1], [3, 3], [6, 6], [4, 3]],
    #                               [[8, 8], [7, 1], [4, 5], [6, 8], [4, 3.5]]])
    # demos_left_x = rf.utils.bezier.multi_bezier_demos(left_demo_points)  # (3, 50, 2): 3 demos, each has 50 points
    # demos_right_x = rf.utils.bezier.multi_bezier_demos(right_demo_points)
    # bi(demos_left_x, demos_right_x, show_demo_idx=2, plot=False)

    # Uni_3d
    demo_points = np.array([[[0, 0, 1], [-1, 8, 1], [4, 3, 2], [2, 1, 2], [4, 3, 2]],
                            [[0, -2, 1], [-1, 7, 1], [3, 2.5, 2], [2, 1.6, 2], [4, 3, 2]],
                            [[0, -1, 1], [-1, 8, 1], [4, 5.2, 2], [2, 1.1, 2], [4, 3.5, 2]]])
    demos_x = rf.utils.bezier.multi_bezier_demos(demo_points)  # (3, 50, 2): 3 demos, each has 50 points
    rf.tpgmm.uni(demos_x, show_demo_idx=2, plot=True)
