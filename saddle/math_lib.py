"""Summary
"""
from typing import Callable, Tuple

import numpy as np

from saddle.errors import OverIterLimitError, PositiveProductError

__all__ = ("ridders_solver", "diagonalize")


def ridders_solver(
    func: Callable[[float], float],
    x1: float,
    x2: float,
    iteration: int = 500,
    error: float = 1e-6,
) -> float:
    """The ridders solver to solver nonlinear equation to find a mathematical
    root for a continuous function. the value of the two end should be of
    different sign.

    Parameters
    ----------
    func : Callable[[float], float]
        function to find the right root
    x1 : float
        left end of interval
    x2 : float
        right end of interval
    iteration : int, optional
        numbers of iterations, default is 30
    error : float, optional
        the threshold for convergence,
        default is 10e-6

    Raises
    ------
    OverIterLimitError
        Description
    PositiveProductError
        when the function value of two ends of the
        interval is of the same sign

    Returns
    -------
    float
        the root for function in the interval between x1 and x2


    """
    f1 = func(x1)
    f2 = func(x2)
    if f1 * f2 > 0:
        raise PositiveProductError("The two end point are of same sign")
    answer: float = 0
    if np.allclose(f1, 0):
        return x1
    elif np.allclose(f2, 0):
        return x2
    for _ in range(iteration):
        x3 = 0.5 * (x1 + x2)
        f3 = func(x3)
        s = np.sqrt(f3 * f3 - f1 * f2)
        if np.allclose(s, 0):
            return answer
        x4 = x3 + (x3 - x1) * np.sign(f1 - f2) * f3 / s
        if abs(x4 - answer) < error:
            return answer
        answer = x4
        f4 = func(x4)
        if np.allclose(f4, 0.0):
            return answer
        if np.sign(f4) != np.sign(f3):
            x1, f1 = x3, f3
            x2, f2 = x4, f4
        elif np.sign(f4) != np.sign(f1):
            x2, f2 = x4, f4
        elif np.sign(f4) != np.sign(f2):
            x1, f1 = x4, f4
        else:
            raise PositiveProductError("The two end point are of same sign")
    raise OverIterLimitError


def diagonalize(
    matrix: "np.ndarray[float]"
) -> Tuple["np.ndarray[float]", "np.ndarray[float]"]:
    """Orthogonilize a given matrix my Grammian Matrix method

    Arguments
    ---------
    matrix : np.ndarray[float]
        Given matrix to be diagonalized

    Returns
    -------
    (w, v) : Tuple['np.ndarray[float]', 'np.ndarray[float]']
        w is the eigenvalues of the Grammian matrix
        v is the eigenvectors of the Grammian matrix, each column is one vector
    """
    product = np.dot(matrix, matrix.T)
    w, v = np.linalg.eigh(product)
    return w, v


def pse_inv(matrix):
    """Calculate pseudo inverse of given matrix

    Parameters
    ----------
    matrix : np.ndarray(N, K)
        a 2-dimention numpy array

    Returns
    -------
    np.ndarray(K, N)
        pseudo inverse of given matrix, inverse if it is revertible
    """
    assert isinstance(matrix, np.ndarray)
    matrix[abs(matrix) < 1e-9] = 0
    shape = matrix.shape[::-1]
    u, s, vh = np.linalg.svd(matrix)
    s[abs(s) < 1e-9] = 0
    s[s != 0] = 1 / s[s != 0]

    s_mtr = np.zeros(shape)
    s_mtr[: len(s), : len(s)] = np.diag(s)
    res = np.dot(np.dot(vh.T, s_mtr), u.T)
    res[abs(res) < 1e-9] = 0
    # infunction test
    diff = np.dot(np.dot(matrix, res), matrix) - matrix
    assert np.allclose(np.linalg.norm(diff), 0), f"pseudo inverse didn't converge"
    return res


def maximum_overlap(target_mtr, input_mtr):
    """Compute the rotation matrix of maximum overlap for given input matrix.

    Parameters
    ----------
    target_mtr : np.ndarray(M, N)
        target basis
    input_mtr : np.ndarray(M, n)
        input basis to be rotate

    Returns
    -------
    np.ndarray
        The transform matrix for input matrix to rotate to maximum overlap
        with target matrix.
    """
    if target_mtr.ndim == 1 or input_mtr.ndim == 1:
        raise ("Input array need to be 2d array, reshape 1d array with (n, 1)")
    if target_mtr.shape != input_mtr.shape:
        raise (
            f"Different shape of matrices, got {target_mtr.shape}, {input_mtr.shape}"
        )
    outer_space = np.dot(target_mtr, input_mtr.T)
    u, _, v = np.linalg.svd(outer_space)
    return np.dot(u, v)
