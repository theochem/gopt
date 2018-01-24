import numpy as np


def modify_hessian(matrix, neg_num, key_ic=0, pos_value=0.05, neg_value=-0.05):
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
            pos_sum = np.sum((pos_vec**2)[:key_ic, :], axis=0)
            seq_ind = np.argsort(pos_sum)[::-1]
            value[seq_ind[:diff] + neg] = neg_value
    else:
        diff = neg - neg_num
        neg_vec = vec[:, :neg]
        if key_ic == total:
            value[neg_num:neg] = pos_value
        else:
            neg_sum = np.sum((neg_vec**2)[:key_ic, :], axis=0)
            seq_ind = np.argsort(neg_sum)
            value[seq_ind[:diff]] = pos_value
    value[(value > 0) & (value < 0.05)] = 0.05
    value[(value > -0.05) & (value < 0)] = -0.05
    return np.dot(np.dot(vec, np.diag(value)), vec.T)


def modify_hessian_with_pos_defi(matrix,
                                 neg_num,
                                 key_ic,
                                 pos_value=0.05,
                                 neg_value=-0.05):
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
