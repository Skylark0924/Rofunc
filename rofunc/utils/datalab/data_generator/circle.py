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


def draw_arc(center, radius, theta1, theta2, color):
    """
    Draw an arc.

    Example::

        >>> from rofunc.utils.datalab.data_generator.circle import draw_arc
        >>> import numpy as np
        >>> xy = draw_arc([-1, 0], 1, 0 * np.pi / 3, 2 * np.pi / 3, color='red')

    :param center: the center of the arc
    :param radius: the radius of the arc
    :param theta1: the start angle of the arc
    :param theta2: the end angle of the arc
    :param color: the color of the arc
    :return: the coordinates of the arc
    """
    # (x-a)²+(y-b)²=r²
    a, b = center
    theta = np.linspace(theta1, theta2, 100)
    x = a + radius * np.cos(theta)
    y = b + radius * np.sin(theta)
    plt.plot(x, y, color=color)

    xy = np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1)))).reshape((1, -1, 2))
    return xy


def draw_connect(l_curve, r_curve, type):
    """
    Draw the connection between two curves.

    :param l_curve: the left curve
    :param r_curve: the right curve
    :param type: line style
    :return:
    """
    for i in range(0, len(l_curve), 10):
        plt.plot([l_curve[i, 0], r_curve[i, 0]], [l_curve[i, 1], r_curve[i, 1]], ls=type, alpha=0.7)
