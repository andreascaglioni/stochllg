from math import sqrt
import numpy as np
import scipy


def compute_inf_sup(B, M_isr, L_isr, type=None):
    """Compute inf-sup constant as minimum positive singular value of an
    appropriate matrix.

    Args:
        B (np.ndarray[float]): Off diagonal matrix in position [0, 1] of saddle
            point system.
        M (np.ndarray[float]): Power -1/2 of matrix representing the scalar
            product of primal variable.
        L (np.ndarray[float]): Power -1/2 of matrix representing the scalar
            product of Lagrange multipliers.
        type (str): "sparse" or "dense" to choose the type of eigenvalue solver.
            Default is None. In this case, a sparse method is chosen for
            matrices with minimum dimension > 10. Dense otherwise.

    Returns:
        float: estimate inf-sup constant
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
    except:
        # print("Warning: Cannot find minimal singular value. Returing NaN")
        return float("nan")


def estimate_inf_sup_const_EIGS(B, M, L):
    """Compute inf-sup constant as sqrt minimum eigenvalue of appropriate
    generalized eigenvalue problem.

    Args:
        B (np.ndarray[float]): Off diagonal matrix in position [1,0] of saddle
            point system
        M (np.ndarray[float]): Matrix representing the scalar product of primal
            variable
        L (np.ndarray[float]): Matrix representing the scalar product of
            Lagrange multipliers

    Returns:
        float: estimate inf-sup constant
    """

    lhs_evp = np.dot(B, np.dot(np.linalg.inv(M), B.T))  # symmetric!
    # Estimate 10 smalles eig.values with method for symmetric matrices
    eigs, _ = scipy.sparse.linalg.eigsh(A=lhs_evp, k=10, M=L, sigma=0.0)
    print("Small eigenvalues (max 10):", eigs[0 : min(10, eigs.size)])
    min_eig = np.amin(eigs)
    assert min_eig > 0
    inf_sup_c = sqrt(min_eig)
    return inf_sup_c
