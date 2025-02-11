import numpy as np

def invert_pseudo(mat, rcond=None):
    """
    Compute the Moore-Penrose pseudo-inverse of a matrix.

    This function computes the pseudo-inverse of a given matrix using 
    Singular Value Decomposition (SVD). If the matrix is invertible, 
    its pseudo-inverse will be equal to its inverse.
    Ref: https://en.wikipedia.org/wiki/Moore–Penrose_pseudoinverse
    
    Parameters:
    ----------
    mat : array_like
        Input matrix (real or complex).
    rcond : float, optional
        Cutoff for small singular values. Singular values smaller 
        than rcond * largest_singular_value are set to zero. 
        If None, NumPy's default value is used.

    Returns:
    -------
    ndarray
        The pseudo-inverse of the input matrix.

    Raises:
    ------
    np.linalg.LinAlgError
        If the SVD computation does not converge.
    TypeError
        If input is not a valid NumPy array or cannot be converted.
    """
    try:
        mat = np.asarray(mat)  # Ensure input is a NumPy array
        if mat.ndim != 2:
            raise ValueError("Input matrix must be 2-dimensional.")

        return np.linalg.pinv(mat, rcond)
    
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(f"SVD computation did not converge: {e}")
    
    except Exception as e:
        raise TypeError(f"Invalid input: {e}")