"""
Parametric expansions of the Wiener process.

This module provides functions to construct the Wiener process using
either a Levy-Ciesielski (LC) or Karhunen-Loeve (KL) expansion.

Functions:
    ``param_LC_W(tt, yy, T)``: Construct Wiener process using LC expansion.
    ``param_KL_Brownian_motion(tt, yy)``: Construct Wiener process using KL expansion.
"""

from math import ceil, log, sqrt, pi
import warnings
import numpy as np


def param_LC_W(yy, tt, T):
    """Pythonic computation of the LC expansion of the Wiener process.

    Args:
        yy (numpy.ndarray[float]): Parameter vector for the expansion.
        tt (numpy.ndarray[float]): 1D array of discrete times in [0, T].
        T (float): Final time of approximation.

    Returns:
        numpy.ndarray[float]: 2D array. Each *ROW* is a sample path of W over tt.
    """

    # Check input shape and make it 2D
    if yy.size == 1 or yy is int:  # 1-element array
        yy = np.array([yy], dtype=float).reshape((1, 1))
    if len(yy.shape) == 1:  # 1 parameter vector
        yy = np.array([yy], dtype=float).reshape((1, yy.size))
    assert len(yy.shape) == 2, (
        "param_LC_Brownian_motion: yy must be 2D (1 ROW per sample array)"
    )
    assert len(tt.shape) == 1, "param_LC_Brownian_motion: tt must be 1D"
    assert np.amin(tt) >= 0 and np.amax(tt) <= T, (
        "param_LC_Brownian_motion: tt not within [0,T]"
    )


    # Get # LC-levels
    L = ceil(log(yy.shape[1], 2))  # levels

    # Extend yy to the next power of 2
    fill = np.zeros((yy.shape[0], 2**L - yy.shape[1]))
    yy = np.column_stack((yy, fill))

    (n_y, dim_y) = yy.shape
    n_t = tt.size

    # Rescale tt (to be reverted!)
    tt = tt / T

    # Compute basis B
    BB = np.zeros((dim_y, n_t))

    BB[0, :] = tt  # first basis function is the linear one
    for lev in range(1, L + 1):
        n_j = 2 ** (lev - 1)  # number of basis functions at level l
        for j in range(1, n_j + 1):
            basis_fun = 0 * tt  # basis is 0 where not assegned below

            # define increasing part basis function
            ran1 = np.where(
                (tt >= (2 * j - 2) / (2**lev)) & (tt <= (2 * j - 1) / (2**lev))
            )
            basis_fun[ran1] = tt[ran1] - (2 * j - 2) / 2**lev

            # define decreasing part basis function
            ran2 = np.where((tt >= (2 * j - 1) / (2**lev)) & (tt <= (2 * j) / (2**lev)))
            basis_fun[ran2] = -tt[ran2] + (2 * j) / 2**lev

            n_b = 2 ** (lev - 1) + j - 1  # prev. lev.s (complete) + curr. lev (partial)
            BB[n_b, :] = 2 ** ((lev - 1) / 2) * basis_fun

    W = np.matmul(yy, BB)

    # Revert rescaling
    W = W * sqrt(T)

    return W


# TODO make yy 2D array and assemble W wih np.prod
def param_LC_W_DEPRECATED(tt, yy, T):
    """
    Construct the Wiener process using the Levy-Ciesielski expansion.

    Args:
        tt (numpy.ndarray[float]): 1D array of discrete times in [0, T].
        yy (numpy.ndarray[float]): Parameter vector for the expansion.
        T (float): Final time of approximation.

    Returns:
        numpy.ndarray[float]: Approximation of the Wiener process at times ``tt``.
    """

    warnings.warn(
        "param_LC_W_DEPRECATED: use param_LC_W instead. This function is deprecated."
    )

    # Check input
    assert np.amin(tt) >= 0, "param_LC_Brownian_motion: tt not within [0,T]"
    assert len(yy.shape) == 1, "param_LC_Brownian_motion: 1 parameter vector at a time"
    if np.any(np.amax(tt) > T) | np.any(np.amin(tt) < 0):
        warnings.warn("Warning...........tt not within [0,T]")

    tt = tt / T  # rescale on [0,1] NB to be reverted below

    # number of levels (nb last level may not have all basis functions!)
    L = ceil(log(len(yy), 2))
    yy = np.append(yy, np.zeros(2**L - len(yy)))  # zero padding to fill level L

    W = yy[0] * tt
    for l in range(1, L + 1):
        for j in range(1, 2 ** (l - 1) + 1):
            eta_n_i = 0 * tt
            # define part of currect basis function corepsonding to (0, 1/2)
            ran1 = np.where((tt >= (2 * j - 2) / (2**l)) & (tt <= (2 * j - 1) / (2**l)))
            eta_n_i[ran1] = tt[ran1] - (2 * j - 2) / 2**l
            # define part of currect basis function corepsonding to (0, 1/2, 1)
            ran2 = np.where((tt >= (2 * j - 1) / (2**l)) & (tt <= (2 * j) / (2**l)))
            eta_n_i[ran2] = -tt[ran2] + (2 * j) / 2**l
            W = W + yy[2 ** (l - 1) + j - 1] * 2 ** ((l - 1) / 2) * eta_n_i

    W = W * np.sqrt(T)  # revert scaling above to go to times in [0,T]

    return W


def param_KL_W(yy, tt):
    """
    Construct the Wiener process using the Karhunen-Loeve expansion.

    Args:
        tt (numpy.ndarray[float]): 1D array of discrete times in [0, 1].
        yy (numpy.ndarray[float]): Parameter vector for the expansion.

    Returns:
        numpy.ndarray[float]: Samples of the Wiener process at times ``tt``.
    """

    W = 0 * tt
    for n in range(len(yy)):
        W = W + pi * (n + 0.5) * sqrt(2) * np.sin((n + 0.5) * pi * tt) * yy[n]
    return W


def sample_W(tt):
    """Sample the Browniann motion with the classical algorithm based on the
    fact that for t_0 < t_1 <= t_2 < t_3,
    - W(t_1)-W(t_0) ~ N(0, t_1-t_0) and
    - W(t_3)-W(t_2) is independent from W(t_2)-W(t_1).

    Args:
        tt (numpy.ndarray[float]): 1D array of time steps. Entries are such that
        tt[i] < tt[i+1] for all i.
        The first entry must be 0.
    """

    # Check input
    assert np.amin(tt) >= 0, "sample_W: tt must be positive"
    assert len(tt) > 1, "sample_W: tt must have at least two entries"
    assert np.all(np.diff(tt) > 0), "sample_W: tt must be strictly increasing"
    assert tt[0] == 0, "sample_W: first entry of tt must be 0"

    # Compute the increments
    W = np.zeros_like(tt)
    for i in range(1, len(tt)):
        W[i] = W[i - 1] + np.random.normal(0, sqrt(tt[i] - tt[i - 1]))

    return W
