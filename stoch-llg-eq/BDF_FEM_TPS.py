"""Module with functions to sample the solution to a parametric LLG equation
coming from the stochastic LLG equation.
"""

from petsc4py import PETSc
import time
import numpy as np
from scipy.special import comb
import ufl
from dolfinx import la
from dolfinx.fem import Constant, Function, form
from ufl import dx, grad, inner, cross, dot
from dolfinx.fem.petsc import assemble_matrix_nest, assemble_vector_nest
from src.inf_sup import compute_inf_sup


def coeffs_bdf(k):
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
    """Compute BDF functions (extrapolation mhat and time derivative past
    information mr) from past magnetizations.

    Args:
        V3 (functionspace): Function space magnetizations
        gamma (list[float]): List of coefficient BDF extrapolation of length bdf_order
        delta (list[float]): List of coefficneti BDF time derivative of length bdf_order+1.
            NB the term delta[0] is never used in this function, but only once v
            (m velocity) is computed to obtain m[j] = mr + tau / delta[0] * v
        mvac_bdf (list[Fuction(V3)]): List of relevant magnetizations of length bdf_order.

    Returns:
        tuple: Tuple with two functions, mhat and mr:
            mhat (Function): BDF extrapolation of magnetization
            mr (Function): BDF time derivative of magnetization
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
    B = A.getNestSubMatrix(1, 0)
    B = B.getValues(range(0, B.getSize()[0]), range(0, B.getSize()[1]))
    beg_time = time.time()
    inf_sup_const = compute_inf_sup(B, ip_V3_isr, ip_V_isr, "sparse")
    end_time = time.time()
    if verb_iter:
        print(f"Inf-sup: {inf_sup_const:.4e}",
              f"(time: {end_time - beg_time:.4f}s)")
    return inf_sup_const


def solve_linear_system(msh, A, b, V3, V, verbose=False):
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
    """This is the same function as high_order_bdf_sllg but with modularized
    code split into several sub-functions.

    Args:
        data (dict): Dictionary with data to solve the problem. The keys are:
            m0h (Function): Initial magnetization, as a V3 function.
            alpha (float): Gilbert damping coefficient.
            gh (Function): Space component noise. Already as a V3 function (see variable V3 below).
            W (numpy.ndarray[float]): A Wiener process sample as an array foe valuation over time steps tt.
            tt (np.ndarray): Time steps.
            bdf_ord (int): BDF order.
            V3 (FunctionSpace): Function space for the magnetization.
            V (FunctionSpace): Function space for the Lagrange multiplier.
        quadrature_degree (int, optional): Quadrature degree. Defaults to 0.
        verbose (bool, optional): Print information. If int, log every verbose interations. If True, log all itnerations. Inf False, run silently. Defaults to False.
        H_input (numpy.ndarray, optional): External magnetic field. Defaults to None.
        compute_inf_sup (bool, optional): Compute inf-sup constant. Defaults to False.
        ip_V_isr (numpy.ndarray[float], optional): Inverse square root of inner product matrix for V. Needed only to compute the inf-sup constant. Defaults to [].
        ip_V3_isr (numpy.ndarray[float], optional): Inverse square root of inner product matrix for V. Needed only to compute the inf-sup constant. Defaults to [].

    Returns:
        list: List of magnetizations.
        list: List of v functions. Shorter than mm by 1.
        list: List of lam functions.  Shorter than mm by 1.
        list: List of inf-sup constants over time steps. Shorter that mm by 1.
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
