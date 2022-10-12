from math import factorial
import numpy as np


# Building piecewise constant basis functions
def build_phi_piecewise(nb_data, nb_fct):
    phi = np.kron(np.identity(nb_fct), np.ones((int(np.ceil(nb_data / nb_fct)), 1)))
    return phi[:nb_data]


# Building radial basis functions (RBFs)
def build_phi_rbf(nb_data, nb_fct):
    t = np.linspace(0, 1, nb_data).reshape((-1, 1))
    tMu = np.linspace(t[0], t[-1], nb_fct)
    phi = np.exp(-1e2 * (t.T - tMu) ** 2)
    return phi.T


# Building Bernstein basis functions
def build_phi_bernstein(nb_data, nb_fct):
    t = np.linspace(0, 1, nb_data)
    phi = np.zeros((nb_data, nb_fct))
    for i in range(nb_fct):
        phi[:, i] = factorial(nb_fct - 1) / (factorial(i) * factorial(nb_fct - 1 - i)) * (1 - t) ** (
                nb_fct - 1 - i) * t ** i
    return phi


# Building Fourier basis functions
def build_phi_fourier(nb_data, nb_fct):
    t = np.linspace(0, 1, nb_data).reshape((-1, 1))

    # Alternative computation for real and even signal
    k = np.arange(0, nb_fct).reshape((-1, 1))
    phi = np.cos(t.T * k * 2 * np.pi)
    return phi.T
