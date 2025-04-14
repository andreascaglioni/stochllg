"""Generate mesh square. Adapted from:
The FEniCSx tutorial
https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html
"""

import numpy as np
from mpi4py import MPI
import gmsh
from dolfinx.io import XDMFFile, gmshio


def generate_gmsh_square(char_length):
    gmsh.initialize()
    # Generate 2D square in 3D
    membrane = gmsh.model.occ.addRectangle(x=0, y=0, z=0, dx=1, dy=1)

    gmsh.model.occ.synchronize()

    gdim = 2  # NB this tells the object is intrinsically 2D!
    gmsh.model.addPhysicalGroup(gdim, [membrane])

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)
    gmsh.model.mesh.generate(gdim)
    return gmsh.model, gdim


################################### MAIN #######################################
if __name__ == "__main__":
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD

    nn_el = 2 ** np.arange(2, 9)  # number of elements per side
    for n_el in nn_el:
        char_length = 1 / n_el
        model, gdim = generate_gmsh_square(char_length=char_length)
        domain, cell_markers, facet_markers = gmshio.model_to_mesh(
            gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim
        )
        xdmf = XDMFFile(
            domain.comm,
            "meshes_square/msh_test_square_" + str(n_el) + ".xdmf",
            "w",
        )
        xdmf.write_mesh(domain)
