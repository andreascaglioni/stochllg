""" Module with functions to sample the solution to a parametric LLG equation
    coming from the stochastic LLG equation.
"""

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import time
import numpy as np
import scipy
import ufl
from dolfinx import la
from dolfinx.fem import Constant, Function, form
from ufl import dx, grad, inner, cross, dot
from dolfinx.fem.petsc import assemble_matrix_nest, assemble_vector_nest
from functions.inf_sup import compute_inf_sup


def coeffs_bdf(k):
    gamma = []
    delta = []
    tmp = 0.0
    for i in range(1, k + 1):
        tmp += 1.0 / i
    delta.append(tmp)
    for i in range(1, k + 1):
        gamma.append(scipy.special.comb(k, i) * (-1) ** (i - 1.0))
        tmp = 0
        for j in range(i, k + 1):
            tmp += scipy.special.comb(j, i) * (-1) ** float(i) / float(j)
        delta.append(tmp)
    return gamma, delta


# TODO Why need j? I always compute the BDF wrt the last array element!
# TODO Call compute_coeffs_BDF inside this function!
# TODO move to ALGEBRAIC opeartions on coordinates vector (efficiency)
# TODO pass only values m on required timesteps
def compute_BDF(V3, gamma, delta, mvec, j, k):
    mhat = Function(V3)  # check if this works: mvec[-1].copy(deepcopy=True)
    mr = Function(V3)
    for i in range(0, k):
        mhat.x.array[:] = mhat.x.array + gamma[i] * mvec[j - (i + 1)].x.array
        mr.x.array[:] = mr.x.array - delta[i + 1] * mvec[j - (i + 1)].x.array
    mr.x.array[:] = mr.x.array / delta[0]
    sq_norm_mhat = np.linalg.norm(mhat.x.array, ord=2)
    mhat.x.array[:] = mhat.x.array / sq_norm_mhat
    return mhat, mr


def assemble_lin_system(
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
    if verbose:
        print("Assembling...", flush=True)
    beg_time = time.time()
    tau_norm = Constant(msh, PETSc.ScalarType(tau / delta[0]))

    jit_opts = {
        "cffi_extra_compile_args": ["-O3", "-march=native"],
        "cffi_libraries": ["m"],
    }

    lhs_eq = form(
        [
            [
                (
                    alpha * inner(v, phi)
                    + inner(cross(mhat, v), phi)
                    + tau_norm
                    * inner(
                        grad(
                            v
                            + Cs * cross(v, gh)
                            + Cc * cross(cross(v, gh), gh)
                        ),
                        grad(
                            phi
                            + Cs * cross(phi, gh)
                            + Cc * cross(cross(phi, gh), gh)
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
                    grad(
                        mr + Cs * cross(mr, gh) + Cc * cross(cross(mr, gh), gh)
                    ),
                    grad(
                        phi
                        + Cs * cross(phi, gh)
                        + Cc * cross(cross(phi, gh), gh)
                    ),
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
        print(end_time - beg_time, "s")
    return A, b


def solve_linear_system(msh, A, b, V3, V, verbose=False):
    if verbose:
        print("Solving linear system...", flush=True)
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
        print(end_time - beg_time, "s")
    return v, lam


def update_m(V3, mr, tau, delta, v):
    m_new = Function(V3)
    m_new.x.array[:] = mr.x.array + tau / delta[0] * v.x.array
    return m_new


def high_order_bdf_sllg_functions(
    bdf_ord,
    alpha,
    T,
    tau,
    m0h,
    V3,
    V,
    W,
    g,
    msh,
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
        k (int): BDF order.
        alpha (float): Damping coefficient.
        T (float): Final time.
        tau (float): Time step.
        minit (Function): Initial magnetization as a V3 Function!!!
        V3 (FunctionSpace): Function space for magnetization.
        V (FunctionSpace): Function space for lam.
        W (numpy.ndarray): Array of angles.
        g (numpy.ndarray): Array of directions.
        quadrature_degree (int, optional): Quadrature degree. Defaults to 0.
        verbose (bool, optional): Print information. Defaults to False.
        H_input (numpy.ndarray, optional): External magnetic field. Defaults to
            None.
        compute_inf_sup (bool, optional): Compute inf-sup constant. Defaults
            to False.
        ip_V_isr (numpy.ndarray[float], optional): Inverse square root of inner
            product matrix for V. Needed only to compute the inf-sup constant.
            Defaults to [].
        ip_V3_isr (numpy.ndarray[float], optional): Inverse square root of inner
            product matrix for V. Needed only to compute the inf-sup constant.
            Defaults to [].

    Returns:
        mm (list): List of magnetizations.
        vv (list): List of v functions. Shorter than mm by 1.
        ll (list): List of lam functions.  Shorter than mm by 1.
        inf_sup_t (list): List of inf-sup constants over time steps. Shorter
            that mm by 1.
    """


    steps = int(T / tau)
    gamma, delta = coeffs_bdf(bdf_ord)
    mm = [m0h]  # in V3
    assert bdf_ord == 1, "Only BDF order 1 implemented"  # TODO implement k>1
    vv = []  # list of functions in V3
    ll = []  # list of functions in V
    inf_sup_t = []

    for j in range(bdf_ord, steps + 1):
        if verbose:
            print("Iteration", j, flush=True)
        mhat, mr = compute_BDF(V3, gamma, delta, mm, j, bdf_ord)
        A, b = assemble_lin_system(
            msh,
            quadrature_degree,
            alpha,
            mhat,
            mr,
            tau,
            delta,
            g,
            W[j],
            V3,
            V,
            H_input,
            verbose,
        )

        if return_inf_sup:
            B = A.getNestSubMatrix(1, 0)
            B = B.getValues(range(0, B.getSize()[0]), range(0, B.getSize()[1]))
            isc = compute_inf_sup(B, ip_V3_isr, ip_V_isr, "sparse")
            inf_sup_t.append(isc)

        v, lam = solve_linear_system(msh, A, b, V3, V, verbose)
        m_new = update_m(V3, mr, tau, delta, v)
        mm.append(Function(V3))
        mm[-1].x.array[:] = m_new.x.array
        vv.append(Function(V3))
        vv[-1].x.array[:] = v.x.array
        ll.append(Function(V))
        ll[-1].x.array[:] = lam.x.array
    return mm, vv, ll, inf_sup_t
