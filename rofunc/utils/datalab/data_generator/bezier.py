#  Copyright (C) 2024, Junjia Liu
# 
#  This file is part of Rofunc.
# 
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
# 
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
# 
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com


import matplotlib.pyplot as plt
import numpy as np
from math import factorial


def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


def get_bezier_curve(points):
    n = len(points) - 1
    return lambda t: sum(comb(n, i) * t ** i * (1 - t) ** (n - i) * points[i] for i in range(n + 1))


def evaluate_bezier(points, total):
    """
    Generate the Bezier curve from points.

    :param points: points for shaping the Bezier curve
    :param total: the number of points on the Bezier curve
    :return:
    """
    bezier = get_bezier_curve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)])
    return new_points


def plot_bezier(bx, by, x, y, bz=None, z=None, ax=None):
    """
    Plot the Bezier curve, either in 2D or 3D.

    Example::

        >>> from rofunc.utils.datalab.data_generator.bezier import plot_bezier
        >>> import numpy as np
        >>> demo_points = np.array([[0, 0], [-1, 8], [4, 3], [2, 1], [4, 3]])
        >>> bx, by = evaluate_bezier(demo_points, 50)[:, 0], evaluate_bezier(demo_points, 50)[:, 1]
        >>> x, y = demo_points[:, 0], demo_points[:, 1]
        >>> plot_bezier(bx, by, x, y)

    :param bx: the x coordinates of the Bezier curve
    :param by: the y coordinates of the Bezier curve
    :param x: the x coordinates of the demonstration points
    :param y: the y coordinates of the demonstration points
    :param bz: the z coordinates of the Bezier curve, default None
    :param z: the z coordinates of the demonstration points, default None
    :param ax: the axis to plot the Bezier curve, default None
    :return:
    """
    assert isinstance(bx, list) or isinstance(bx, np.ndarray)
    if bz is not None and z is not None:
        for i in range(len(bx)):
            ax.plot(bx[i], by[i], bz[i])
            # ax.plot(x[i], y[i], z[i], 'r.')
    else:
        for i in range(len(bx)):
            plt.plot(bx[i], by[i])
            # plt.plot(x[i], y[i], 'r.')


def multi_bezier_demos(demo_points, ax=None):
    """
    Generate multiple Bezier curves as demonstrations.

    Example::

        >>> from rofunc.utils.datalab.data_generator.bezier import multi_bezier_demos
        >>> import numpy as np
        >>> demo_points = np.array([[[0, 0], [-1, 8], [4, 3], [2, 1], [4, 3]],
        ...                         [[0, -2], [-1, 7], [3, 2.5], [2, 1.6], [4, 3]],
        ...                         [[0, -1], [-1, 8], [4, 5.2], [2, 1.1], [4, 3.5]]])
        >>> demos_x = multi_bezier_demos(demo_points)

    :param demo_points: points for shaping the Bezier curve
    :param ax: the axis to plot the Bezier curves, default None
    :return:
    """
    bx_lst = [evaluate_bezier(demo_point, 50)[:, 0] for demo_point in demo_points]
    by_lst = [evaluate_bezier(demo_point, 50)[:, 1] for demo_point in demo_points]
    x_lst = [demo_point[:, 0] for demo_point in demo_points]
    y_lst = [demo_point[:, 1] for demo_point in demo_points]

    if len(demo_points[0][0]) == 3:
        bz_lst = [evaluate_bezier(demo_point, 50)[:, 2] for demo_point in demo_points]
        z_lst = [demo_point[:, 2] for demo_point in demo_points]
        plot_bezier(bx_lst, by_lst, x_lst, y_lst, bz_lst, z_lst, ax)
        demos_x = np.concatenate(
            (np.array(bx_lst).reshape((len(demo_points), -1, 1)), np.array(by_lst).reshape((len(demo_points), -1, 1)),
             np.array(bz_lst).reshape((len(demo_points), -1, 1))), axis=2)
    else:
        plot_bezier(bx_lst, by_lst, x_lst, y_lst)
        demos_x = np.concatenate(
            (np.array(bx_lst).reshape((len(demo_points), -1, 1)), np.array(by_lst).reshape((len(demo_points), -1, 1))),
            axis=2)
    return demos_x
