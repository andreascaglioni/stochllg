"""Utility functions for finite element computations and error analysis.

This module provides a collection of utility functions for working with finite element
spaces, exporting data, computing errors, and handling mesh-related operations.

Functions:
    ``compute_rate(xx, yy)``:
        Compute the rate of change between two arrays.

    ``export_xdmf(msh, f, tt=np.array([]), filename="plot.xdmf")``:
        Export finite element functions to an XDMF file.

    ``get_H1_matrix(V3)``:
        Compute the H1 inner product matrix for a given finite element space.

    ``get_L2_matrix(V)``:
        Compute the L2 inner product matrix for a given finite element space.

    ``ip_norm(x, A=None)``:
        Compute the norm of a vector with respect to an inner product matrix.

    ``compute_data_nonmatch_interpol(V_exa, V)``:
        Compute interpolation data for non-matching finite element spaces.

    ``error_space_time(u_exa, tt_exa, U_in, tt, ip_matrix, matching_x_spaces=True, data_nonmatch=None, t_error_type="L2")``:
        Compute the space-time error between two functions.

    ``inverse_sqrt(A)``:
        Compute the inverse square root of a matrix.

    ``float_f(x)``:
        Format a float variable in scientific notation.

    ``mesh_elems_area(msh)``:
        Compute the area of elements in a 2D mesh.

    ``set_FE_data(msh, data)``:
        Set finite element data for a given mesh and save it into a dictionary.
"""

from math import sqrt
from copy import deepcopy
import numpy as np
from scipy.linalg import eigh
from scipy.interpolate import interp1d

# Import dolfinx
from dolfinx import default_real_type
import ufl
from dolfinx.io import XDMFFile
from dolfinx.fem import (
    Function,
    form,
    create_interpolation_data,
    element,
    functionspace,
    Expression
)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from ufl import dx, grad, inner, TrialFunction, TestFunction
from basix.ufl import element


def compute_rate(xx, yy):
    """
    Compute the logarithmic rate of change between consecutive elements of two arrays.

    Args:
        xx (numpy.ndarray): 1D array of positive x-coordinates.
        yy (numpy.ndarray): 1D array of positive y-coordinates.

    Returns:
        numpy.ndarray: Logarithmic rates of change.

    Raises:
        ValueError: If arrays have different lengths or contain non-positive values.

    Example:
        >>> compute_rate(np.array([1, 2, 4]), np.array([2, 4, 8]))
        array([1., 1.])
    """
    return np.log(yy[1:] / yy[:-1]) / np.log(xx[1:] / xx[:-1])


def export_xdmf(msh, f, tt=np.array([]), filename="plot.xdmf"):
    """
    Exports a mesh and associated functions to an XDMF file.

    Args:
        msh (Mesh): The mesh to export.
        f (Function or list of Function): The function(s) to export.
        tt (numpy.ndarray, optional): Time steps for the functions. Defaults to an empty array.
        filename (str, optional): Name of the output XDMF file. Defaults to "plot.xdmf".

    Raises:
        TypeError: If `f` is not a Function or a list of Functions.
    """
    xdmf = XDMFFile(msh.comm, filename, "w")
    xdmf.write_mesh(msh)
    if type(f) is list and type(f[0]) is Function:
        if tt.size == 0:
            Warning("export_xdmf: Missing time tt. Using 1,2,...")
            tt = np.linspace(0, len(f) - 1, len(f))
        # export in sequence
        for i in range(len(f)):
            f[i].name = "f"
            xdmf.write_function(f[i], tt[i])
    elif type(f) is Function:
        f.name = "f"
        xdmf.write_function(f)
    else:
        raise TypeError("f has unknown type for export")
    xdmf.close()


def get_H1_matrix(V3):
    """
    Compute the symmetrized H1 inner product matrix for a finite element space.

    Args:
        V3 (FunctionSpace): Finite element function space.

    Returns:
        numpy.ndarray: Symmetrized H1 inner product matrix.
    """

    v_trial = TrialFunction(V3)
    v_test = TestFunction(V3)
    H1_product_form = form(
        (inner(v_trial, v_test) + inner(grad(v_trial), grad(v_test))) * dx
    )
    H1_product = assemble_matrix(H1_product_form)
    H1_product.assemble()
    sz = H1_product.size
    H1_product = H1_product.getValues(range(0, sz[0]), range(0, sz[1]))
    # symmetrize
    H1_product = 0.5 * (H1_product + H1_product.T)
    return H1_product


def get_L2_matrix(V):
    l_trial = TrialFunction(V)
    l_test = TestFunction(V)
    L2_product_form = form(inner(l_test, l_trial) * dx)
    L2_product = assemble_matrix(L2_product_form)
    L2_product.assemble()
    sz = L2_product.size
    L2_product = L2_product.getValues(range(0, sz[0]), range(0, sz[1]))
    # symmetrize
    L2_product = 0.5 * (L2_product + L2_product.T)
    return L2_product


def ip_norm(x, A=None):
    """
    Compute the norm of a vector with respect to an inner product matrix.

    Args:
        x (numpy.ndarray): Input vector.
        A (numpy.ndarray, optional): Inner product matrix. Defaults to the identity matrix.

    Returns:
        float: Norm of the vector.
    """
    if A is None:
        A = np.eye(x.size)  # Euclidean inner product
    return np.sqrt(np.dot(x, np.dot(A, x)))


# TODO implement time interpolation
# TODO what if higher degree finite elements spaces? Interpolation still working?
def compute_data_nonmatch_interpol(V_exa, V):
    """
    Compute interpolation data for non-matching finite element spaces.

    Args:
        V_exa (FunctionSpace): Exact/reference finite element space.
        V (FunctionSpace): Approximation finite element space.

    Returns:
        tuple: Cells and interpolation data for non-matching spaces.
    """
    mesh_exa = V_exa.mesh
    mesh_exa_cell_map = mesh_exa.topology.index_map(mesh_exa.topology.dim)
    num_cells_on_proc = mesh_exa_cell_map.size_local + mesh_exa_cell_map.num_ghosts
    cells = np.arange(num_cells_on_proc, dtype=np.int32)
    interpolation_data = create_interpolation_data(V_exa, V, cells)
    return cells, interpolation_data


# def error_unit_modulus(mm):
#     n_tt = len(mm)
#     errmagtime = np.zeros(n_tt)
#     V = mm[0].function_space.sub(0).collapse()[0]  # collapse() returns tuple. Second eleemnt is a map from old to new
#     error_function = Function(V)
#     for i in range(n_tt):
#         m = mm[i]
#         m_mag = ufl.sqrt(m[0] ** 2 + m[1] ** 2 + m[2] ** 2)  # in l2 or Euclidean norm
#         error_function.interpolate(Expression(1-m_mag, V.element.interpolation_points()))
        
#         errmagtime[i] = sqrt(asseinner(error_function, error_function) * ufl.dx)
    # return errmagtime

def error_space_time(
    u_exa,
    tt_exa,
    U_in,
    tt,
    ip_matrix,
    matching_x_spaces=True,
    data_nonmatch=None,
    t_error_type="L2",
):
    """Compute error of two functions in space-time.

    Args:
        u_exa (list[Function]): First function (the exact or reference one)
        tt_exa (numpy.array[float]): Array of time steps of u_exa.
        U_in (list[Function]): Second function (the approximation of u_exa)
        dtdt (numpy.array[float]): Array of time step sizes. NB its length is ff.size-1!
        ip_matrix (numpy.array[float]): Square matrix represnting inner product in space of exact solution.
        matching_x_spaces (bool): If True, the spaces for the x variable of u_exa and U_in are matching. Defaults to True.
        data_nonmatch ([tuple]): Tuple (cells, interpolation_data) needed to call ``interpolate_nonmatching``. Defaults to []. In this case, the data is computed. NB this is needed only if reference/exact and approximation spaces are not matching! Can be computed with ``compute_data_nonmatch_interpol``.

    Return:
        tuple[float, numpy.ndarray[float]]: Tuple of: The (non-negative) error; The error in space at each time step.
    """

    # Ensure time steps sizes are coherent
    assert len(u_exa) == tt_exa.size
    assert len(U_in) == tt.size
    assert tt_exa.size >= tt.size

    # If spaces are NOT matching, need additional data for interpolation of U into the space of u_exa
    V_exa = u_exa[0]._V
    V = U_in[0]._V
    if not matching_x_spaces:
        if data_nonmatch is None:  # if no data is provided, compute it now
            data_nonmatch = compute_data_nonmatch_interpol(V_exa, V)
        cells, interpolation_data = data_nonmatch

    # Guarantee that U and u_exa are defined on the same time steps tt_exa
    tt_different = tt.size < tt_exa.size or (
        tt.size == tt_exa.size and np.linalg.norm(tt - tt_exa) > 1.e-10
    )
    if tt_different:
        U_intdata = np.array([U_in[i].x.array for i in range(len(U_in))])
        interpolant_U_t = interp1d(tt, U_intdata, kind="linear", axis=0)
        U_dofs = interpolant_U_t(tt_exa)
    else:
        U_dofs = np.array([U_in[i].x.array for i in range(len(U_in))])

    # Compute space error for each time step (interpolate U(t_i) in V_exa)
    f_Vexa = Function(V_exa)
    f_V = Function(V)
    err_tt = np.zeros(len(u_exa))
    for i in range(len(u_exa)):
        # Guarantee that U and U_exa are in same FE space
        if not matching_x_spaces:
            f_V.x.array[:] = U_dofs[i]
            f_Vexa.interpolate_nonmatching(f_V, cells, interpolation_data)
        elif V == V_exa:  # the FE spaces are the same
            f_Vexa.x.array[:] = U_dofs[i]
        else:  # the FE spaces are matching but not the same
            f_V.x.array[:] = U_dofs[i]
            f_Vexa.interpolate(f_V)
        # Compute error in FE space
        f_Vexa.x.array[:] -= u_exa[i].x.array
        err_tt[i] = ip_norm(f_Vexa.x.array, A=ip_matrix)

    # Compute space+time error
    dtdt = tt_exa[1:] - tt_exa[:-1]
    if t_error_type == "L2":
        err = sqrt(np.sum(dtdt * (err_tt[1:] ** 2)))
    elif t_error_type == "L1":
        err = np.sum(dtdt * err_tt)
    elif t_error_type == "Linf":
        err = np.amax(err_tt)
    else:
        raise ValueError("Unknown time error type")

    return err, err_tt


def inverse_sqrt(A):
    """
    Compute the inverse square root of a matrix.

    Args:
        A (numpy.ndarray): Input matrix.

    Returns:
        numpy.ndarray: Inverse square root of the matrix.
    """
    e_val, e_vec = eigh(A)
    return e_vec @ np.diag(1.0 / np.sqrt(e_val)) @ e_vec.T


def float_f(x):
    """
    Format a float variable in scientific notation.

    Args:
        x (float): Input float.

    Returns:
        str: Formatted float as a string.
    """
    return f"{x:.4e}"


def mesh_elems_area(msh):
    """Compute the area of the elements of a 2D mesh.

    Args:
        msh (dolfinxmesh.Mesh): Mesh

    Returns:
        np.ndarray([float]): Size of each element in the mesh.
    """

    DG0 = functionspace(msh, ("DG", 0))
    v = ufl.TestFunction(DG0)
    cell_area_form = form(v * ufl.dx)
    cell_area = assemble_vector(cell_area_form)
    return cell_area.array


def set_FE_data(msh, data):
    """
    Set finite element data for a given mesh and save it into a dictionary.

    Args:
        msh (dolfinx.mesh.Mesh): Input mesh.
        data (dict): Dictionary containing problem data.

    Returns:
        dict: Updated dictionary with finite element data.
    """
    Pr = element(
        "Lagrange", msh.basix_cell(), data["fem_order"], dtype=default_real_type
    )
    Pr3 = element(
        "Lagrange",
        msh.basix_cell(),
        data["fem_order"],
        shape=(3,),
        dtype=default_real_type,
    )
    V = functionspace(msh, Pr)
    V3 = functionspace(msh, Pr3)
    m0h = Function(V3)
    m0h.interpolate(data["m0"])
    gh = Function(V3)
    gh.interpolate(data["g"])

    # Deep-copy data into new dictionary and add FE data
    data_out = deepcopy(data)
    data_out["m0h"] = m0h
    data_out["gh"] = gh
    data_out["msh"] = msh
    data_out["V"] = V
    data_out["V3"] = V3
    data_out["ip_V3"] = get_H1_matrix(V3)
    data_out["ip_V"] = get_L2_matrix(V)
    return data_out
