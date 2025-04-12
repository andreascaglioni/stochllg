import ufl
import dolfinx.fem
def mesh_elems_area(msh):
    """Compute the size of the elements in a 2D mesh.

    Args:
        msh (dolfinxmesh.Mesh): Mesh
    
    Returns:
        np.ndarray([float]): Size of each element in the mesh.
    """

    DG0 = dolfinx.fem.functionspace(msh, ("DG", 0))
    v = ufl.TestFunction(DG0)
    cell_area_form = dolfinx.fem.form(v*ufl.dx)
    cell_area = dolfinx.fem.assemble_vector(cell_area_form)
    return cell_area.array
