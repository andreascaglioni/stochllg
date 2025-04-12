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


# TODO make yy 2D array and assemble W wih np.prod
def param_LC_W(tt, yy, T):
    """
    Construct the Wiener process using the Levy-Ciesielski expansion.

    Args:
        tt (numpy.ndarray[float]): 1D array of discrete times in [0, T].
        yy (numpy.ndarray[float]): Parameter vector for the expansion.
        T (float): Final time of approximation.

    Returns:
        numpy.ndarray[float]: Approximation of the Wiener process at times ``tt``.
    """

    # Check input
    assert np.amin(tt) >= 0, "param_LC_Brownian_motion: tt not within [0,T]"
    assert len(yy.shape) == 1, "param_LC_Brownian_motion: 1 parameter vector at a time"
    if np.any(np.amax(tt) > T) | np.any(np.amin(tt) < 0):
        warnings.warn("Warning...........tt not within [0,T]")
    
    tt = tt/T  # rescale on [0,1] NB to be reverted below

    # number of levels (nb last level may not have all basis functions!)
    L = ceil(log(len(yy), 2))
    yy = np.append(yy, np.zeros(2**L-len(yy)))  # zero padding to fill level L

    W = yy[0] * tt
    for l in range(1, L + 1):
        for j in range(1, 2 ** (l - 1) + 1):
            eta_n_i = 0 * tt
            # define part of currect basis function corepsonding to (0, 1/2)
            ran1 = np.where(\
                (tt >= (2 * j - 2) / (2 ** l)) & (tt <= (2 * j - 1) / (2 ** l))\
                    )
            eta_n_i[ran1] = tt[ran1] - (2 * j - 2) / 2 ** l
            # define part of currect basis function corepsonding to (0, 1/2, 1)
            ran2 = np.where(\
                (tt >= (2 * j - 1) / (2 ** l)) & (tt <= (2 * j) / (2 ** l))\
                    )
            eta_n_i[ran2] = - tt[ran2] + (2 * j) / 2 ** l
            W = W + yy[2 ** (l - 1) + j - 1] * 2 ** ((l - 1) / 2) * eta_n_i

    W = W*np.sqrt(T)  # revert scaling above to go to times in [0,T]

    return W

def param_KL_Brownian_motion(tt, yy):
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
        W = W + pi*(n+0.5)*sqrt(2)*np.sin((n+0.5)*pi*tt)*yy[n]
    return W
