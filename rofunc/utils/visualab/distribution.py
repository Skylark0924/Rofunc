import matplotlib.pyplot as plt
import numpy as np
import rofunc as rf
import scipy as sp
from tqdm import tqdm


def gmm_plot2d(Mu, Sigma, nbStates, color=[1, 0, 0], alpha=0.5, linewidth=1, markersize=6,
               ax=None, empty=False, edgecolor=None, edgealpha=None, priors=None,
               border=False, nb=1, swap=True, center=True, zorder=20):
    """
    This function displays the parameters of a Gaussian Mixture Model (GMM).

     Inputs -----------------------------------------------------------------
       o Mu:           D x K array representing the centers of K Gaussians.
       o Sigma:        D x D x K array representing the covariance matrices of K Gaussians.

     Author:    Martijn Zeestraten, 2015
             http://programming-by-demonstration.org/martijnzeestraten

             Note- Daniel Berio, switched matrix layout to be consistent with pbdlib matlab,
                   probably breaks with gmm now.
    """
    nbDrawingSeg = 35
    t = np.linspace(-np.pi, np.pi, nbDrawingSeg)

    for i, c, a in zip(range(0, nbStates), color, alpha):
        # Create Polygon
        if not swap:
            R = np.real(sp.linalg.sqrtm(1.0 * Sigma[:, :, i]))
            points = R.dot(
                np.array([[np.cos(t)], [np.sin(t)]]).reshape([2, nbDrawingSeg])) + Mu[:,
                                                                                   i].reshape(
                [2, 1])
        else:
            R = np.real(sp.linalg.sqrtm(1.0 * Sigma[i]))
            points = R.dot(np.array([[np.cos(t)], [np.sin(t)]]).reshape([2, nbDrawingSeg])) + \
                     Mu[[i]].T

        if edgecolor is None:
            edgecolor = c

        if priors is not None: a *= priors[i]

        polygon = plt.Polygon(points.transpose().tolist(), facecolor=c, alpha=a,
                              linewidth=linewidth, zorder=zorder, edgecolor=edgecolor)

        if edgealpha is not None:
            plt.plot(points[0, :], points[1, :], color=edgecolor)

        if nb == 2:
            R = np.real(sp.linalg.sqrtm(4.0 * Sigma[:, :, i]))
            points = R.dot(np.array([[np.cos(t)], [np.sin(t)]]).reshape([2, nbDrawingSeg])) + Mu[:, i].reshape([2, 1])
            polygon_2 = plt.Polygon(points.transpose().tolist(), facecolor=c, alpha=a / 2.,
                                    linewidth=linewidth, zorder=zorder - 5, edgecolor=edgecolor)
        # Set properties
        # polygon.set_alpha(0.3)
        # polygon.set_color(color)
        if ax:
            if nb == 2:
                ax.add_patch(polygon_2)
            ax.add_patch(polygon)  # Patch

            l = None
            if center:
                a = alpha[i]
            else:
                a = 0.

            if not swap:
                ax.plot(Mu[0, i], Mu[1, i], '.', color=c, alpha=a)  # Mean
            else:
                ax.plot(Mu[i, 0], Mu[i, 1], '.', color=c, alpha=a)  # Mean

            if border:
                ax.plot(points[0, :], points[1, :], color=c, linewidth=linewidth,
                        markersize=markersize)  # Contour
        else:
            if empty:
                plt.gca().grid('off')
                # ax[-1].set_xlabel('x position [m]')
                plt.gca().set_axis_bgcolor('w')
                plt.axis('off')

            plt.gca().add_patch(polygon)  # Patch
            if nb == 2:
                ax.add_patch(polygon_2)
            l = None

            if center:
                a = alpha[i]
            else:
                a = 0.0

            if not swap:
                l, = plt.plot(Mu[0, i], Mu[1, i], '.', color=c, alpha=a)  # Mean
            else:
                l, = plt.plot(Mu[i, 0], Mu[i, 1], '.', color=c, alpha=a)  # Mean

            if border:
                plt.plot(points[0, :], points[1, :], color=c, linewidth=linewidth,
                         markersize=markersize)  # Contour
            # plt.plot(points[0,:], points[1,:], color=c, linewidth=linewidth , markersize=markersize) # Contour

    return l


def gmm_plot3d(mu, covariance, color, alpha=0.5, ax=None, scale=0.1):
    """
    plots points and their corresponding gmm model in 3D
    Input:
        points: N X 3, sampled points
        w: n_gaussians, gmm weights
        mu: 3 X n_gaussians, gmm means
        stdev: 3 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    """
    n_gaussians = mu.shape[0]
    # Visualize data
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        plt.title('3D GMM')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(35.246, 45)
    plt.set_cmap('Set1')
    for i in tqdm(range(0, n_gaussians, 10)):
        # Plot the ellipsoid
        R = np.real(sp.linalg.sqrtm(scale * covariance[i, :]))
        rf.visualab.sphere_plot3d(mu[i, :], R, color=color, alpha=alpha, ax=ax)
        # Plot the center
        ax.plot(mu[i, 0], mu[i, 1], mu[i, 2], '.', color=color, alpha=1)  # Mean


def gmm_plot(Mu, Sigma, dim=None, color=[1, 0, 0], alpha=0.5, linewidth=1, markersize=6,
             ax=None, empty=False, edgecolor=None, edgealpha=None, priors=None,
             border=False, nb=1, swap=True, center=True, zorder=20, scale=0.2):
    Mu = np.array(Mu)
    Sigma = np.array(Sigma)
    if Mu.ndim == 1:
        if not swap:
            Mu = Mu[:, np.newaxis]
            Sigma = Sigma[:, :, np.newaxis]
            nbStates = 1
        else:
            Mu = Mu[np.newaxis]
            Sigma = Sigma[np.newaxis]
            nbStates = 1
    else:
        if not swap:
            nbStates = Mu.shape[1]
            nbVar = Mu.shape[0]
        else:
            nbStates = Mu.shape[0]
            nbVar = Mu.shape[1]

    if dim:
        if swap:
            Mu = Mu[:, dim]
            sl = np.ix_(range(nbStates), dim, dim)
            Sigma = Sigma[sl]
        else:
            Mu = Mu[dim, :]
            sl = np.ix_(dim, dim)
            Sigma = Sigma[sl]

    if priors is not None:
        priors /= np.max(priors)
        priors = np.clip(priors, 0.1, 1.)

    if len(dim) == 2:
        if not isinstance(color, list) and not isinstance(color, np.ndarray):
            color = [color] * nbStates
        elif not isinstance(color[0], str) and not isinstance(color, np.ndarray):
            color = [color] * nbStates

        if not isinstance(alpha, np.ndarray):
            alpha = [alpha] * nbStates
        else:
            alpha = np.clip(alpha, 0.1, 0.9)
        gmm_plot2d(Mu, Sigma, nbStates, color, alpha, linewidth, markersize,
                   ax, empty, edgecolor, edgealpha, priors, border, nb, swap, center, zorder)
    elif len(dim) > 2:
        gmm_plot3d(Mu, Sigma, color, alpha, ax, scale)
    else:
        raise Exception('Dimension is less than 2, cannot plot')


if __name__ == '__main__':
    # from sklearn.mixture import GaussianMixture

    means = np.array([[0.5, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [-0.5, -0.5, -0.5],
                      [-0.8, 0.3, 0.4]])
    # # covs = np.array([[0.25471843, 0.10747855, 0.17343172, 0.10100491],
    # #                  [0.09279714, 0.1013973, 0.257998, 0.21407591],
    # #                  [0.09945191, 0.17013928, 0.09957319, 0.09709307]]).T
    covs = np.array([np.diag([0.01, 0.01, 0.03]),
                     np.diag([0.08, 0.01, 0.01]),
                     np.diag([0.01, 0.05, 0.01]),
                     np.diag([0.03, 0.07, 0.01])])
    # n_gaussians = means.shape[0]
    # points = []
    # for i in range(len(means)):
    #     x = np.random.multivariate_normal(means[i], covs[i], 1000)
    #     points.append(x)
    # points = np.concatenate(points)
    #
    # # fit the gaussian model
    # gmm = GaussianMixture(n_components=n_gaussians, covariance_type='full')
    # gmm.fit(points)
    #
    # gmm_plot3d(means.T, covs.T, [0, 0, 0, 0])

    # means = np.array([0.5, 0.0, 0.0])
    # covs = np.diag([6, 12, 0.1])
    # eigen_value, eigen_vector = np.linalg.eig(covs)
    # radii = np.sqrt(eigen_value)
    gmm_plot3d(means, covs)
