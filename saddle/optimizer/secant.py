import numpy as np
from saddle.math_lib import pse_inv
from saddle.optimizer.path_point import PathPoint


def secant(new_ob, old_ob):  # need tests
    assert isinstance(new_ob, PathPoint)
    assert isinstance(old_ob, PathPoint)
    delta_g = new_ob.v_gradient - old_ob.v_gradient
    delta_v = new_ob.vspace - old_ob.vspace
    delta_b = new_ob.b_matrix - old_ob.b_matrix
    part1 = np.dot(np.dot(new_ob.b_matrix.T, delta_v),
                   new_ob.v_gradient)  # v_space gradient here is cartesian
    part2 = np.dot(delta_b.T, new_ob.q_gradient)  # gradient here is internal
    inv_trans = np.dot(new_ob.vspace.T, pse_inv(new_ob.b_matrix.T))
    result = delta_g - np.dot(inv_trans, (part1 + part2))
    return result
