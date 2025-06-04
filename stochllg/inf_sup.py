"""
Inf-sup constant computation.

This module provides functions to compute the inf-sup constant for saddle point systems.

Functions:
    - ``compute_inf_sup(B, M_isr, L_isr, type=None)``: Estimate the inf-sup constant using singular values.
    - ``estimate_inf_sup_const_EIGS(B, M, L)``: Estimate the inf-sup constant using eigenvalues.

"""

from math import sqrt
import numpy as np
import scipy


def compute_inf_sup(B, M_isr, L_isr, type=None):
    """
    Estimate the inf-sup constant using singular values.

    Args:
        B (np.ndarray[float]): Off-diagonal matrix of the saddle point system.
        M_isr (np.ndarray[float]): Inverse square root of the primal scalar product matrix.
        L_isr (np.ndarray[float]): Inverse square root of the Lagrange multipliers' scalar product matrix.
        type (str, optional): "sparse" or "dense" SVD solver. Defaults to "sparse" for large matrices.

    Returns:
        float: Estimated inf-sup constant.
    """

    if type is None:
        type = "sparse" if min(B.shape) > 20 else "dense"
    
    B2 = np.dot(L_isr, np.dot(B, M_isr))
    try:
        if type == "sparse":
            s_vals = scipy.sparse.linalg.svds(
                B2, k=5, which="SM", return_singular_vectors=False
            )
        elif type == "dense":
            s_vals = scipy.linalg.svd(
                B2, full_matrices=False, compute_uv=False
            )
        else:
            raise ValueError("Unknown type:", type, "for SVD algorithm.")
        return np.amin(s_vals)
    except Exception as e:
        print("Error during SVD computation:", e)
        print("Cannot compute minimal singular value. Returing NaN")
        return float("nan")


def estimate_inf_sup_const_EIGS(B, M_inverse, L, verbose=False):
    """
    Estimate the inf-sup constant using eigenvalues.

    Args:
        B (np.ndarray[float]): Off-diagonal matrix of the saddle point system.
        M (np.ndarray[float]): Inverse of primal scalar product matrix.
        L (np.ndarray[float]): Lagrange multipliers scalar product matrix.

    Returns:
        float: Estimated inf-sup constant.
    """
    # M_inverse = np.linalg.inv(M)
    lhs_evp = np.dot(B, np.dot(M_inverse, B.T))  # symmetric!
    
    # Estimate 10 smalles eig.values with method for symmetric matrices
    eigs, _ = scipy.sparse.linalg.eigsh(A=lhs_evp, k=10, M=L, sigma=0.0)
    if verbose:
        print("Smallest eigenvalues (max 10):", eigs[0 : min(10, eigs.size)])
    
    min_eig = np.amin(eigs)
    assert min_eig > 0, f"Error: Minimum eigenvalue is negative: {min_eig}"
    inf_sup_c = sqrt(min_eig)
    
    return inf_sup_c
