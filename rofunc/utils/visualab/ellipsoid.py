# Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import nestle
import numpy as np
from matplotlib import cm
from numpy import linalg

import rofunc as rf


def sphere_plot3d(mean, cov, color=[1, 0, 0], alpha=0.2, ax=None):
    """
    Plot 3D sphere or ellipsoid

    Example::

        >>> import rofunc as rf
        >>> import numpy as np
        >>> from matplotlib import pyplot as plt
        >>> means = np.array([0.5, 0.0, 0.0])
        >>> covs = np.diag([6, 12, 0.1])
        >>> rf.visualab.sphere_plot3d(means, covs)
        >>> plt.show()

    :param mean: the mean point coordinate of sphere
    :param cov: the covariance matrix of the sphere
    :param color: the color of the ellipsoid
    :param alpha: the transparency of the ellipsoid
    :param ax: the axis to plot the ellipsoid
    """
    # ell_gen = nestle.Ellipsoid(mean, np.dot(cov.T, cov))
    ell_gen = nestle.Ellipsoid(mean, np.linalg.inv(cov))
    npoints = 100
    points = ell_gen.samples(npoints)
    pointvol = ell_gen.vol / npoints
    # Find bounding ellipsoid(s)
    ells = nestle.bounding_ellipsoids(points, pointvol)

    # plot
    if ax is None:
        fig = plt.figure(figsize=(10., 10.))
        ax = fig.add_subplot(111, projection='3d')
    for ell in ells:
        plot_ellipsoid_3d(ell, ax, color, alpha)


def plot_ellipsoid_3d(ell, ax, color, alpha):
    """
    Plot the 3-d Ellipsoid ell on the Axes3D ax.

    :param ell: the ellipsoid to plot
    :param ax: the axis to plot the ellipsoid
    :param color: the color of the ellipsoid
    :param alpha: the transparency of the ellipsoid
    """
    # points on unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    z = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    x = np.outer(np.ones_like(u), np.cos(v))

    # transform points to ellipsoid
    for i in range(len(x)):
        for j in range(len(x)):
            x[i, j], y[i, j], z[i, j] = ell.ctr + np.dot(ell.axes, [x[i, j], y[i, j], z[i, j]])

    ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, alpha=alpha)


def ellipsoid_plot3d(ellipsoids, mode='quaternion', Rs=None):
    """
    Plot the ellipsoids in 3d, used by `rf.robolab.manipulability.mpb.get_mpb_from_model`

    :param ellipsoids: list of ellipsoids to plot
    :param mode: 'quaternion' or 'euler' or 'given'
    :param Rs: rotation matrices
    :return: None
    """
    if mode == 'given':
        assert Rs is not None, "Rotation matrices are not given"
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # set colour map so each ellipsoid as a unique colour
    n_ellip = len(ellipsoids)
    norm = colors.Normalize(vmin=0, vmax=n_ellip)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # compute each and plot each ellipsoid iteratively
    for index in range(n_ellip):
        # your ellipsoid and center in matrix form
        if mode in ['quaternion', 'euler']:
            center = ellipsoids[index, :3]

            if mode == 'quaternion':
                R = rf.robolab.coord.homo_matrix_from_quaternion(ellipsoids[index, 3:7])
            elif mode == 'euler':
                R = rf.robolab.coord.homo_matrix_from_euler(ellipsoids[index, 3], ellipsoids[index, 4],
                                                            ellipsoids[index, 5],
                                                            'sxyz')

            # find the rotation matrix and radii of the axes
            U, s, rotation = linalg.svd(R)
            radii = 1.0 / np.sqrt(s) * 0.3  # reduce radii by factor 0.3

            # calculate cartesian coordinates for the ellipsoid surface
            u = np.linspace(0.0, 2.0 * np.pi, 60)
            v = np.linspace(0.0, np.pi, 60)
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        elif mode == 'given':
            center = np.zeros(3)

            rotation = Rs[index]
            radii = ellipsoids[index]
            # calculate cartesian coordinates for the ellipsoid surface
            u = np.linspace(0.0, 2.0 * np.pi, 60)
            v = np.linspace(0.0, np.pi, 60)
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        else:
            raise ValueError("Unknown mode")

        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

        ax.plot_surface(x, y, z, rstride=3, cstride=3, color=m.to_rgba(index), linewidth=0.1, alpha=0.2, shade=True)

    # rf.visualab.set_axis(ax)
    plt.show()
