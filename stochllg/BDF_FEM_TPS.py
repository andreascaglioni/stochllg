"""
A high-order Tangent Plane Scheme for sampling (in time and space) the 
stochastic Landau-Lifshitz-Gilbert (LLG) equation.

This algorithm uses:
- Backward Differentiation Formula (BDF) for time discretization.
- Finite Element Method (FEM) for spatial discretization.
- Tangent Plane Scheme (TPS) for enforcing the orthogonality constraint between trial/test spaces and the magnetization itself.

Functions:

- ``coeffs_bdf(k)``: Compute BDF coefficients for extrapolation and time derivative.
- ``compute_BDF(V3, gamma, delta, mvac_bdf)``: Compute BDF extrapolation and time derivative.
- ``_assemble_lin_system(...)``: Assemble the linear system for the LLG equation.
- ``inf_sup(A, ip_V_isr, ip_V3_isr, verb_iter)``: Compute the inf-sup constant.
- ``solve_linear_system(msh, A, b, V3, V, verbose)``: Solve the linear system.
- ``update_m(V3, mr, tau, delta, v)``: Update the magnetization.
- ``BDF_FEM_TPS(...)``: Solve the parametric LLG equation using BDF-FEM-TPS.

"""

from petsc4py import PETSc
import time
import numpy as np
from scipy.special import comb

# Import dolfinx and ufl
import ufl
from dolfinx import la
from dolfinx.fem import Constant, Function, form
from ufl import dx, grad, inner, cross, dot
from dolfinx.fem.petsc import assemble_matrix_nest, assemble_vector_nest

# Import from this project
from stochllg.inf_sup import compute_inf_sup


def coeffs_bdf(k):
    """
    Compute BDF coefficients for extrapolation and time derivative.

    Args:
        k (int): BDF order.

    Returns:
        tuple: Coefficients for extrapolation (gamma) and time derivative (delta).
    """
    gamma = []  # k coefficient for BDF extrapolation
    delta = []  # k+1 coefficient for BDF time derivative
    tmp = 0.0
    for i in range(1, k + 1):
        tmp += 1.0 / i
    delta.append(tmp)
    for i in range(1, k + 1):
        gamma.append(comb(k, i) * (-1) ** (i - 1.0))
        tmp = 0
        for j in range(i, k + 1):
            tmp += comb(j, i) * (-1) ** float(i) / float(j)
        delta.append(tmp)
    return gamma, delta


def compute_BDF(V3, gamma, delta, mvac_bdf):
    """
    Compute BDF extrapolation and time derivative from past magnetizations.

    Args:
        V3 (FunctionSpace): Function space for magnetizations.
        gamma (list[float]): Coefficients for BDF extrapolation.
        delta (list[float]): Coefficients for BDF time derivative.
        mvac_bdf (list[Function]): Past magnetizations.

    Returns:
        tuple: Extrapolated magnetization (mhat) and time derivative (mr).
    """
    assert len(mvac_bdf) == len(gamma), "Wrong number of BDF coefficients"
    assert len(mvac_bdf) == len(delta) - 1, "Wrong number of BDF coefficients"

    bdf_ord = len(mvac_bdf)
    mhat = Function(V3)
    mr = Function(V3)
    for i in range(0, bdf_ord):
        mhat.x.array[:] = mhat.x.array + gamma[i] * mvac_bdf[bdf_ord - 1 - i].x.array
        mr.x.array[:] = mr.x.array - delta[i + 1] * mvac_bdf[bdf_ord - 1 - i].x.array
    mr.x.array[:] = mr.x.array / delta[0]
    sq_norm_mhat = np.linalg.norm(mhat.x.array, ord=2)
    mhat.x.array[:] = mhat.x.array / sq_norm_mhat
    return mhat, mr


def _assemble_lin_system(
    msh,
    quad_deg,
    alpha,
    mhat,
    mr,
    tau,
    delta,
    gh,
    W_j,
    V3,
    V,
    H_input=None,
    verbose=False,
):
    """
    Assemble the linear system for the LLG equation.

    Args:
        ...: Various inputs including mesh, coefficients, and function spaces.

    Returns:
        tuple: Assembled matrix (A) and vector (b).
    """
    (v, lam) = ufl.TrialFunction(V3), ufl.TrialFunction(V)
    (phi, mu) = ufl.TestFunction(V3), ufl.TestFunction(V)

    # avoid just-in-time (JIT) compilation at every timestep with Constant
    Cs = Constant(msh, PETSc.ScalarType(np.sin(W_j)))
    Cc = Constant(msh, PETSc.ScalarType(1 - np.cos(W_j)))

    # build external magnetic field
    if H_input is not None:
        H = Constant(H_input)
    else:
        H = Constant(
            msh,
            (
                PETSc.ScalarType(0.0),
                PETSc.ScalarType(0.0),
                PETSc.ScalarType(0.0),
            ),
        )
    HH = -Cs * cross(H, gh) + Cc * cross(cross(H, gh), gh)

    # define LLG form
    if quad_deg == 0:
        dxr = dx
    else:
        dxr = dx(metadata={"quadrature_degree": quad_deg})

    beg_time = time.time()
    tau_norm = Constant(msh, PETSc.ScalarType(tau / delta[0]))

    jit_opts = {
        "cffi_extra_compile_args": ["-O3", "-march=native"],
        "cffi_libraries": ["m"],
    }

    # TODO: split definition in several variables to make readable
    lhs_eq = form(
        [
            [
                (
                    alpha * inner(v, phi)
                    + inner(cross(mhat, v), phi)
                    + tau_norm
                    * inner(
                        grad(v + Cs * cross(v, gh) + Cc * cross(cross(v, gh), gh)),
                        grad(
                            phi + Cs * cross(phi, gh) + Cc * cross(cross(phi, gh), gh)
                        ),
                    )
                )
                * dxr,
                inner(dot(phi, mhat), lam) * dxr,
            ],
            [inner(dot(v, mhat), mu) * dxr, None],
        ],
        jit_options=jit_opts,
    )

    rhs_eq = form(
        [
            (
                -inner(
                    grad(mr + Cs * cross(mr, gh) + Cc * cross(cross(mr, gh), gh)),
                    grad(phi + Cs * cross(phi, gh) + Cc * cross(cross(phi, gh), gh)),
                )
            )
            * dxr,
            inner(Constant(msh, PETSc.ScalarType(0)), mu) * dx,
        ],
        jit_options=jit_opts,
    )
    # Assebly
    A = assemble_matrix_nest(lhs_eq)
    A.assemble()
    b = assemble_vector_nest(rhs_eq)
    end_time = time.time()
    if verbose:
        print(f"Assembly time: {end_time - beg_time:.4f}s")
    return A, b


def inf_sup(A, ip_V_isr, ip_V3_isr, verb_iter):
    """
    Compute the inf-sup constant for the linear system.

    Args:
        A (Matrix): Assembled matrix.
        ip_V_isr (np.ndarray): Inverse square root of inner product matrix for V.
        ip_V3_isr (np.ndarray): Inverse square root of inner product matrix for V3.
        verb_iter (bool): Verbosity flag.

    Returns:
        float: Inf-sup constant.
    """
    B = A.getNestSubMatrix(1, 0)
    B = B.getValues(range(0, B.getSize()[0]), range(0, B.getSize()[1]))
    beg_time = time.time()
    inf_sup_const = compute_inf_sup(B, ip_V3_isr, ip_V_isr, "sparse")
    end_time = time.time()
    if verb_iter:
        print(f"Inf-sup: {inf_sup_const:.4e}", f"(time: {end_time - beg_time:.4f}s)")
    return inf_sup_const


def solve_linear_system(msh, A, b, V3, V, verbose=False):
    """
    Solve the linear system for the LLG equation.

    Args:
        msh (Mesh): Mesh object.
        A (Matrix): Assembled matrix.
        b (Vector): Assembled vector.
        V3 (FunctionSpace): Function space for magnetization.
        V (FunctionSpace): Function space for Lagrange multipliers.
        verbose (bool, optional): Verbosity flag. Defaults to False.

    Returns:
        tuple: Solutions for magnetization (v) and Lagrange multipliers (lam).
    """
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.setFromOptions()
    # ksp.setTolerances(rtol=1e-9)

    v, lam = Function(V3), Function(V)
    x = PETSc.Vec().createNest(
        [la.create_petsc_vector_wrap(v.x), la.create_petsc_vector_wrap(lam.x)]
    )
    beg_time = time.time()
    ksp.solve(b, x)
    end_time = time.time()
    if verbose:
        print(f"Solve time: {end_time - beg_time:.4f}s")
    return v, lam


def update_m(V3, mr, tau, delta, v):
    """
    Update the magnetization using the BDF scheme.

    Args:
        V3 (FunctionSpace): Function space for magnetization.
        mr (Function): Time derivative of magnetization.
        tau (float): Time step size.
        delta (list[float]): BDF coefficients.
        v (Function): Solution vector.

    Returns:
        Function: Updated magnetization.
    """
    m_new = Function(V3)
    m_new.x.array[:] = mr.x.array + tau / delta[0] * v.x.array
    return m_new


def BDF_FEM_TPS(
    data,
    quadrature_degree=0,
    verbose=False,
    H_input=None,
    return_inf_sup=False,
    ip_V_isr=[],
    ip_V3_isr=[],
):
    """
    Solve the parametric Landau-Lifshitz-Gilbert (LLG) equation using the 
    Backward Differentiation Formula (BDF), Finite Element Method (FEM), 
    and Tangential Projection Scheme (TPS).

    Args:
        data (dict): A dictionary containing the problem setup and parameters:

            - **m0h** (*Function*): Initial magnetization function.
            - **alpha** (*float*): Damping parameter.
            - **gh** (*Function*): External field function.
            - **W** (*np.ndarray*): 1D array of values of Brownian motion on time steps.
            - **tt** (*np.ndarray*): Array of time steps.
            - **bdf_order** (*int*): Order of the BDF scheme (currently only 1 is supported).
            - **msh** (*Mesh*): The computational mesh.
            - **V3** (*FunctionSpace*): Function space for vector fields.
            - **V** (*FunctionSpace*): Function space for scalar fields.
            
        quadrature_degree (*int*, optional): Degree of quadrature used for numerical integration. 
            Defaults to 0, which uses the default quadrature degree.
        verbose (*bool* or *int*, optional): Controls verbosity of the output:
            - If `False`, no output is printed.
            - If `True`, detailed output is printed for every iteration.
            - If an integer, output is printed every `verbose` iterations.
            Defaults to `False`.
        H_input (*Function*, optional): Optional input for an external magnetic field. 
            If `None`, no external field is applied. Defaults to `None`.
        return_inf_sup (*bool*, optional): If `True`, computes and returns the inf-sup constants 
            for the linear systems solved at each time step. Defaults to `False`.
        ip_V_isr (*list*, optional): List of inverse square root inner products for the scalar 
            function space *V*. Used for inf-sup constant computation. Defaults to an empty list.
        ip_V3_isr (*list*, optional): List of inverse square root inner products for the vector 
            function space *V3*. Used for inf-sup constant computation. Defaults to an empty list.

    Returns:
        tuple: A tuple containing:
            - *list[Function]*: Magnetization functions at each time step.
            - *list[Function]*: Velocity functions at each time step (excluding the initial step).
            - *list[Function]*: Lagrange multiplier functions at each time step (excluding the initial step).
            - *np.ndarray[float]*: Array of inf-sup constants for each time step (if `return_inf_sup` is `True`).
    """
    # Handle verbosity: turn into int
    if verbose is True:  # log everything
        print_freq = 1
    elif type(verbose) is int:
        print_freq = verbose
    else:  # False or unkonw verbosity value
        print_freq = 0

    # Unpack data dictionary
    m0h = data["m0h"]
    alpha = data["alpha"]
    gh = data["gh"]
    W = data["W"]
    tt = data["tt"]
    bdf_order = data["bdf_order"]
    msh = data["msh"]
    V3 = data["V3"]
    V = data["V"]

    assert bdf_order == 1, "Only BDF order 1 implemented"  # TODO implement k>1

    n_tt = tt.size
    gamma, delta = coeffs_bdf(bdf_order)

    # TODO do not store a list of Functions, rather 1 function (IC) and DOFS
    mm = [Function(V3) for _ in range(n_tt)]  # coordinates magnetization
    mm[0].x.array[:] = m0h.x.array
    vv = [Function(V3) for _ in range(n_tt - 1)]
    ll = [Function(V) for _ in range(n_tt - 1)]
    inf_sup_t = np.zeros(n_tt - 1)
    for j in range(bdf_order, n_tt):
        # verbosity this iteration
        verb_iter = (print_freq > 0) and (j % print_freq == 0)

        if verb_iter:
            print("Iteration", j, flush=True)

        tau = tt[j] - tt[j - 1]

        mhat, mr = compute_BDF(V3, gamma, delta, mm[j - bdf_order : j])

        A, b = _assemble_lin_system(
            msh,
            quadrature_degree,
            alpha,
            mhat,
            mr,
            tau,
            delta,
            gh,
            W[j],
            V3,
            V,
            H_input,
            verb_iter,
        )

        if return_inf_sup:
            inf_sup_t[j - 1] = inf_sup(A, ip_V_isr, ip_V3_isr, verb_iter)

        v, lam = solve_linear_system(msh, A, b, V3, V, verb_iter)
        m_new = update_m(V3, mr, tau, delta, v)

        mm[j].x.array[:] = m_new.x.array
        vv[j - 1].x.array[:] = v.x.array
        ll[j - 1].x.array[:] = lam.x.array
    return mm, vv, ll, inf_sup_t
