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


import numpy as np
from math import factorial


def build_phi_piecewise(nb_data, nb_fct):
    """
    Build piecewise constant basis functions.

    :param nb_data: number of data points
    :param nb_fct: number of basis functions
    :return: phi
    """
    phi = np.kron(np.identity(nb_fct), np.ones((int(np.ceil(nb_data / nb_fct)), 1)))
    return phi[:nb_data]


def build_phi_rbf(nb_data, nb_fct):
    """
    Build radial basis functions (RBFs).

    :param nb_data: number of data points
    :param nb_fct: number of basis functions
    :return: the
    """
    t = np.linspace(0, 1, nb_data).reshape((-1, 1))
    tMu = np.linspace(t[0], t[-1], nb_fct)
    phi = np.exp(-1e2 * (t.T - tMu) ** 2)
    return phi.T


def build_phi_bernstein(nb_data, nb_fct):
    """
    Build Bernstein basis functions.

    :param nb_data: number of data points
    :param nb_fct: number of basis functions
    :return: phi
    """
    t = np.linspace(0, 1, nb_data)
    phi = np.zeros((nb_data, nb_fct))
    for i in range(nb_fct):
        phi[:, i] = factorial(nb_fct - 1) / (factorial(i) * factorial(nb_fct - 1 - i)) * (1 - t) ** (
                nb_fct - 1 - i) * t ** i
    return phi


def build_phi_fourier(nb_data, nb_fct):
    """
    Build Fourier basis functions.

    :param nb_data: number of data points
    :param nb_fct: number of basis functions
    :return: phi
    """
    t = np.linspace(0, 1, nb_data).reshape((-1, 1))

    # Alternative computation for real and even signal
    k = np.arange(0, nb_fct).reshape((-1, 1))
    phi = np.cos(t.T * k * 2 * np.pi)
    return phi.T
