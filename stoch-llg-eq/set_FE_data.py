from copy import deepcopy
from src.utils import get_H1_matrix, get_L2_matrix
from basix.ufl import element
from dolfinx import default_real_type
from dolfinx.fem import Function, functionspace


def set_FE_data(msh, data):
    """Compute and save the mesh-dependent data into the data dictionary."""

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
    m0h.interpolate(lambda x: data["m0"](x))
    gh = Function(V3)
    gh.interpolate(lambda x: data["g"](x))
    
    # Deep-copy data inot new dictionary and add FE data
    data_out = deepcopy(data)
    data_out["m0h"] = m0h
    data_out["gh"] = gh
    data_out["msh"] = msh
    data_out["V"] = V
    data_out["V3"] = V3
    data_out["ip_V3"] = get_H1_matrix(V3)
    data_out["ip_V"] = get_L2_matrix(V)
    return data_out