#!/usr/bin/python3
# !--*-- coding: utf-8 --*--
import numpy as np
import matplotlib.pyplot as plt
from math import factorial


def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


def get_bezier_curve(points):
    n = len(points) - 1
    return lambda t: sum(comb(n, i) * t ** i * (1 - t) ** (n - i) * points[i] for i in range(n + 1))


def evaluate_bezier(points, total):
    bezier = get_bezier_curve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)])
    return new_points[:, 0], new_points[:, 1]


def plot_bezier(bx, by, x, y):
    if type(bx) is list:
        for i in range(len(bx)):
            plt.plot(bx[i], by[i])
            plt.plot(x[i], y[i], 'r.')
        # plt.show()
    else:
        plt.plot(bx, by, 'b-')
        plt.plot(x, y, 'r.')
        plt.show()


def multi_bezier_demos(demo_points):
    bx_lst, by_lst, x_lst, y_lst = [], [], [], []
    for demo_point in demo_points:
        x, y = demo_point[:, 0], demo_point[:, 1]
        bx, by = evaluate_bezier(demo_point, 50)
        bx_lst.append(bx)
        by_lst.append(by)
        x_lst.append(x)
        y_lst.append(y)
    plt.ylim((-2.5, 8.5))
    plot_bezier(bx_lst, by_lst, x_lst, y_lst)
    demos_x = np.concatenate(
        (np.array(bx_lst).reshape((len(demo_points), -1, 1)), np.array(by_lst).reshape((len(demo_points), -1, 1))),
        axis=2)
    return demos_x


if __name__ == '__main__':
    points = np.array([[0, 0], [-1, 8], [4, 3], [2, 1], [4, 3]])
    x, y = points[:, 0], points[:, 1]
    bx, by = evaluate_bezier(points, 50)

    plot_bezier(bx, by, x, y)
