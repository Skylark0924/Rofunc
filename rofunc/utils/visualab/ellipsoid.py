import rofunc as rf
import numpy as np
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors


def ellipsoid_plot3d(ellipsoids):
    """
    Plot the ellipsoids in 3d
    Args:
        ellipsoids: list or array including several 7-dim pose (3 for center, 4 for (w, x, y, z), )

    """
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
        center = ellipsoids[index, :3]
        R = rf.coord.quaternion_matrix(ellipsoids[index, 3:7])

        # find the rotation matrix and radii of the axes
        U, s, rotation = linalg.svd(R)
        radii = 1.0 / np.sqrt(s) * 0.3  # reduce radii by factor 0.3

        # calculate cartesian coordinates for the ellipsoid surface
        u = np.linspace(0.0, 2.0 * np.pi, 60)
        v = np.linspace(0.0, np.pi, 60)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

        ax.plot_surface(x, y, z, rstride=3, cstride=3, color=m.to_rgba(index), linewidth=0.1, alpha=0.5, shade=True)
    plt.show()


if __name__ == '__main__':
    # TODO
    ...
