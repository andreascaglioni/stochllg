"""Generate a seuqence of structured unit square meshes with the included method dolfinx.mesh.create_unit_square"""

from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx.io import XDMFFile
import numpy as np


nn = 4*2 ** np.arange(7)  # number of elements on the side
for n in nn:
    print("n elements per side=", n)
    filename = "msh_square/mesh_square_" + str(int(n)) + ".xdmf"
    msh = create_unit_square(MPI.COMM_SELF, n, n)
    xdmf = XDMFFile(msh.comm, filename, "w")
    xdmf.write_mesh(msh)
    xdmf.close()
