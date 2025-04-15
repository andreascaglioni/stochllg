"""Define data for an example on the convergence of the TPS with respect to dt."""

from math import pi
from os.path import join
import numpy as np

from mpi4py import MPI
from dolfinx.io import XDMFFile
from stochllg.utils import set_FE_data
from stochllg.parametric_W import param_LC_W

# Physics data
alpha = 1.4
T = 1


def W_fun(tt, yy):  # Wiener process
    return param_LC_W(tt, yy, T=T)


def m0(x):  # IC
    C = 0.9
    m00 = C * (x[0] - 0.5)
    m01 = C * (x[1] - 0.5)
    m02 = np.sqrt(1.0 - np.square(m00) - np.square(m01))
    return np.stack((m00, m01, m02))


def g(x):  # space component noise
    sqr = np.square(x[0]) + np.square(x[1])
    C = 0.6
    g0 = C * np.sin(0.5 * pi * sqr) * (x[0]-0.5)
    g1 = C * np.sin(0.5 * pi * sqr) * (x[1]-0.5)    
    g2 = np.sqrt(1.0 - np.square(g0) - np.square(g1))
    return np.stack((g0, g1, g2))


# Discretization space and time
bdf_order = 1
mesh_filename = join("data", "mesh_square_structured", "mesh_square_32.xdmf")
fem_order = 1
comm = MPI.COMM_SELF

# Finite elements data
with XDMFFile(comm, mesh_filename, "r") as xdmf:
    msh = xdmf.read_mesh()

# No time stepping data. To be added at run-time

data = {
    "m0": m0,
    "g": g,
    "fem_order": fem_order,
    "bdf_order": bdf_order,
    "T": T,
    "alpha": alpha,
    "W_fun": W_fun,
}

data = set_FE_data(msh, data)
