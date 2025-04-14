"""Data for h-converngece TPS. Data related to finite elements from mesh and
discretization is removed. The data is set with function set_FE_data() at run time."""

from math import pi
import numpy as np
from mpi4py import MPI
from stochllg.parametric_W import param_LC_W

# Physics data
alpha = 1.4
T = 1


def W_fun(t, y):  # Wiener process
    return 0.*param_LC_W(t, y, T=T)


def m0(x):  # IC
    d = (x[0]-0.5)**2+(x[1]-0.5)**2
    check = (d <= 0.25)
    m00 = np.where(check, x[0]-0.5, 0.)
    m01 = np.where(check, x[1]-0.5, 0.)
    m02 = np.sqrt(1.0 - np.square(m00) - np.square(m01))
    return np.stack((m00, m01, m02))


def g(x):  # space component noise
    sqr = np.square(x[0]) + np.square(x[1])
    C = 0.6
    g0 = C * np.sin(0.5 * pi * sqr) * x[0]
    g1 = C * np.sin(0.5 * pi * sqr) * x[1]
    g2 = np.sqrt(1.0 - np.square(g0) - np.square(g1))
    return np.stack((g0, g1, g2))


# Discretization space and time
fem_order = 1
bdf_order = 1
comm = MPI.COMM_SELF

# FE data removed because computed at run-time

# Time stepping data
tau = 5.0e-3
n_tt = int(T / tau) + 1
tt = np.linspace(0, T, n_tt)

data = {
    "alpha": alpha,
    "m0": m0,
    "g": g,
    "W_fun": W_fun,
    "tt": tt,
    "bdf_order": bdf_order,
    "fem_order": fem_order,
    # "msh": None,
    # "V3": None,
    # "V": None,
    # "ip_V3": None,
    # "ip_V": None,
    # "m0h": None,
    # "gh": None,
}
