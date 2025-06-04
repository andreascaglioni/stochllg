"""A simple test for the inf-sup computation through singular values."""

from os.path import join
import numpy as np
from mpi4py import MPI
import sys

from dolfinx import fem, io, default_real_type
import ufl
import basix


sys.path.insert(0, "./")  # Import from this project
from stochllg.inf_sup import estimate_inf_sup_const_EIGS
from stochllg.utils import get_H1_matrix, get_L2_matrix, float_f
from stochllg.utils import mesh_elems_area as mea


def compute_infsup_tps_sys(msh, fem_order, m):
    Pr = basix.ufl.element(
        "Lagrange", msh.basix_cell(), fem_order, dtype=default_real_type
    )
    Pr3 = basix.ufl.element(
        "Lagrange",
        msh.basix_cell(),
        fem_order,
        shape=(3,),
        dtype=default_real_type,
    )
    V = fem.functionspace(msh, Pr)
    V3 = fem.functionspace(msh, Pr3)
    # Assemble B
    mh = fem.Function(V3)
    mh.interpolate(m)
    v = ufl.TrialFunction(V3)
    mu = ufl.TestFunction(V)
    b = ufl.inner(ufl.dot(v, mh), mu) * ufl.dx
    B = fem.petsc.assemble_matrix(fem.form(b))
    B.assemble()
    B = B.getValues(range(0, B.getSize()[0]), range(0, B.getSize()[1]))
    # Assemble isq scalar products
    ip_V = get_L2_matrix(V)
    # ip_V_isr = inverse_sqrt(ip_V)
    ip_V3 = get_H1_matrix(V3)
    # ip_V3_isr = inverse_sqrt(ip_V3)
    
    # Compute inf-sup
    ip_V3_inv = np.linalg.inv(ip_V3)
    isc = estimate_inf_sup_const_EIGS(B, ip_V3_inv, ip_V)
    return isc


if __name__ == "__main__":
    # SETTINGS
    comm = MPI.COMM_SELF
    np.set_printoptions(formatter={"float_kind": float_f})

    def m(x):
        # return np.stack((0.0 * x[0], 0.0 * x[0], 0.0 * x[0] + 1.0))
        C = 0.9
        m00 = C * (x[0] - 0.5)
        m01 = C * (x[1] - 0.5)
        m02 = np.sqrt(1.0 - np.square(m00) - np.square(m01))
        return np.stack((m00, m01, m02))
    fem_order = 1
    iidx = [4, 8, 16, 32]
    for idx in iidx:
        mesh_filename = join("data", "mesh_square_structured", f"mesh_square_{idx}.xdmf")
        with io.XDMFFile(comm, mesh_filename, "r") as xdmf:
            msh = xdmf.read_mesh()
        h = np.sqrt(np.amin(mea(msh)))
        isc = compute_infsup_tps_sys(msh, fem_order, m)
        print("h:", float_f(h), "ISC:", float_f(isc))
        exit()