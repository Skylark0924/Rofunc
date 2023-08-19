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
