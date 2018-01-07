import numpy as np


def modify_hessian(matrix,
                   *_,
                   neg_num,
                   key_ic,
                   pos_value=0.05,
                   neg_value=-0.05):
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
        pos_sum = np.sum((pos_vec**2)[:key_ic, :], axis=0)
        seq_ind = np.argsort(pos_sum)[::-1]
        value[seq_ind[:diff] + neg] = neg_value
    else:
        diff = neg - neg_num
        neg_vec = vec[:, :neg]
        neg_sum = np.sum((neg_vec**2)[:key_ic, :], axis=0)
        seq_ind = np.argsort(neg_sum)
        value[seq_ind[:diff]] = pos_value
    value[(value > 0) & (value < 0.05)] = 0.05
    value[(value > -0.05) & (value < 0)] = -0.05
    return np.dot(np.dot(vec, np.diag(value)), vec.T)
