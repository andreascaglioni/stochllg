"""Generate a sequence of structured/unstructured unit square meshes with
- dolfinx.mesh.create_unit_square:structured mesh
- gmsh: unstructured mesh *
The meshes are saved in XDMF format.

* Adapted from the FEniCSx tutorial https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html
"""

from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx.io import XDMFFile
import numpy as np
import gmsh
from dolfinx.io import XDMFFile, gmshio


def generate_structured_mesh_square_fenics(comm, n):
    """Generate a structured mesh for a unit square using FEniCS-x and save it to a file.

        comm (MPI communicator): The MPI communicator to use for parallel processing.
        n (int): The number of divisions along each axis of the unit square.
        
    Returns:
        None"""

    return create_unit_square(comm, n, n)
    


def generate_unstruct_mesh_square_gmsh(comm, char_length):
    gmsh_model_rank = 0
    gmsh.initialize()
    membrane = gmsh.model.occ.addRectangle(x=0, y=0, z=0, dx=1, dy=1)
    gmsh.model.occ.synchronize()
    gdim = 2  # NB this tells the object is intrinsically 2D!
    gmsh.model.addPhysicalGroup(gdim, [membrane])
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)
    gmsh.model.mesh.generate(gdim)
    domain, _, _ = gmshio.model_to_mesh(gmsh.model, comm, gmsh_model_rank, gdim=gdim)
    return domain


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    nn = 2 ** np.arange(2, 5)  # number of elements per side
    FUN = "fenics"
    
    for n in nn:
        print("n elements per edge:", n)
        
        if FUN == "fenics":
            msh = generate_structured_mesh_square_fenics(comm, n)
        elif FUN == "gmsh":
            msh = generate_unstruct_mesh_square_gmsh(comm, char_length=1./n)

        # Export mesh to XDMF
        filename = "msh_square/mesh_square_" + str(int(n)) + ".xdmf"
        xdmf = XDMFFile(comm, filename, "w")
        xdmf.write_mesh(msh)