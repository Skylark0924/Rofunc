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
