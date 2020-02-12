"""Modify matrix engenvalues module."""
import numpy as np


def modify_hessian(matrix, neg_num, key_ic=0, pos_value=0.005, neg_value=-0.005):
    """Modify eigenvalues of given matrix.

    Parameters
    ----------
    matrix : np.ndarray(N, N)
        A symmetric square matrix with well defined eigenvalues
    neg_num : int
        number of negative eigenvalue(s) preferred.
    key_ic : int, optional
        number of key internal coordinates in the system
    pos_value : float, optional
        the positive eigenvalues threshold
    neg_value : float, optional
        the negative eigenvalues threshold

    Returns
    -------
    np.ndarray(N, N)
        New modified square matrix
    """
    val, vec = np.linalg.eigh(matrix)
    value = val.copy()
    total = len(value)
    assert 0 <= neg_num <= total
    assert 0 <= key_ic <= total
    neg = np.sum([value < 0])
    if neg == neg_num:
        value[:neg][value[:neg] > neg_value] = neg_value
        value[neg:][value[neg:] < pos_value] = pos_value
    elif neg < neg_num:
        diff = neg_num - neg
        pos_vec = vec[:, neg:]
        if key_ic == total:
            value[neg:neg_num] = neg_value
        else:
            pos_sum = np.sum((pos_vec ** 2)[:key_ic, :], axis=0)
            seq_ind = np.argsort(pos_sum)[::-1]
            value[seq_ind[:diff] + neg] = neg_value
    else:
        diff = neg - neg_num
        neg_vec = vec[:, :neg]
        if key_ic == total:
            value[neg_num:neg] = pos_value
        else:
            neg_sum = np.sum((neg_vec ** 2)[:key_ic, :], axis=0)
            seq_ind = np.argsort(neg_sum)
            value[seq_ind[:diff]] = pos_value
    value[(value > 0) & (value < pos_value)] = pos_value
    value[(value > neg_value) & (value < 0)] = neg_value
    return np.dot(np.dot(vec, np.diag(value)), vec.T)


def modify_hessian_with_pos_defi(
    matrix, neg_num, key_ic, pos_value=0.005, neg_value=-0.005
):
    """Modify hessian matrix with non reduced part positive definite.

    Parameters
    ----------
    matrix : np.ndarray(N, N)
        the original matrix to be modified
    neg_num : int
        number of negative eigenvalues
    key_ic : int
        number of key internal coordinates
    pos_value : float, optional
        positive eigenvalues threshold
    neg_value : TYPE, optional
        negative eigenvalues threshold

    Returns
    -------
    np.ndarray(N, N)
        New modified square matrix
    """
    assert neg_num <= key_ic
    matrix = matrix.copy()
    assert len(matrix[:, 0]) == len(matrix[0])  # make sure it is square
    out_mat = matrix[key_ic:, key_ic:]
    in_mat = matrix[:key_ic, :key_ic]
    out_part_mat = modify_hessian(out_mat, 0, 0, pos_value)
    matrix[key_ic:, key_ic:] = out_part_mat
    in_part_mat = modify_hessian(in_mat, neg_num, key_ic, pos_value, neg_value)
    matrix[:key_ic, :key_ic] = in_part_mat
    # out_part_mat[:key_ic, :key_ic] = in_part_mat
    # matrix[key_ic:, key_ic:] = out_part_mat
    return modify_hessian(matrix, neg_num, key_ic, pos_value, neg_value)
