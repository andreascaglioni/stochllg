import numpy as np


def profit_sllg_fast(nu, p=2):
    """Compute the sparse grid profits of given multi-indices.
    For piecewise-polynomials of degree p-1 and parametric SLLG.

    Args:
        nu (numpy.ndarray[float]): 2D array, each row is a multi-index.
        p (int): Integer >= 2. Piecewise polynomial interpolation degree + 1.

    Returns:
        numpy.ndarray[float]: 1D array of (non-negative) profits. Length equals number of rows of nu.
    """

    #  Check input
    if not (isinstance(p, int) and p > 1):
        raise TypeError("p is not int >= 2")
    if not isinstance(nu, np.ndarray):
        raise TypeError("nu must be a numpy.ndarray.")
    if nu.ndim != 2:
        raise ValueError("nu must be a 2D array.")
    if not np.issubdtype(nu.dtype, int):
        raise TypeError("nu must be an array of int.")
    if np.any(nu < 0):
        raise ValueError("All entries of nu must be non-negative.")

    N_nu, D = nu.shape
    C1, C2 = 1, 1

    w1 = np.asarray(nu == 1).nonzero()
    w2 = np.asarray(nu > 1).nonzero()

    # Levels Levy-Ciesielski expansion (same shape as nu)
    nn = np.arange(1, D + 1, 1)  # linear indices mids
    ell = np.ceil(np.log2(nn))  # log-indices mids
    ell = np.reshape(ell, (1, ell.size))
    ell = np.repeat(ell, N_nu, axis=0)

    # The regularity weights (radii of C-disks of holomorphic extension)
    rho = np.zeros_like(nu, dtype=float)
    if not (w1[0].size == 0):  # if entry 0 is empty, the other is too
        rho[w1] = 2.0 ** (3.0 / 2.0 * ell[w1])
    if not (w2[0].size == 0):
        rho[w2] = 2.0 ** (1.0 / 2.0 * ell[w2])

    # Compute value
    V_comps = np.ones_like(rho)
    V_comps[w1] = C1 * np.power(rho[w1], -1.0)
    V_comps[w2] = C2 * np.power(2.0, -p * nu[w2] * rho[w2])
    value = np.prod(V_comps, axis=1)

    # Compute work
    W_comps = (2 ** (nu + 1) - 2) * (p - 1) + 1
    work = np.prod(W_comps, axis=1)

    return value / work
