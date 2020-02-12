"""Different secant condition method module."""
import numpy as np

from saddle.math_lib import pse_inv
from saddle.optimizer.path_point import PathPoint


def secant(new_ob, old_ob):  # need tests
    """Compute secant value with method mentioned in the original thesis.

    Parameters
    ----------
    new_ob : PathPoint
        Previous step point
    old_ob : PathPoint
        Newly computed step point

    Returns
    -------
    np.ndarray(3N)
        computed secont condition value
    """
    assert isinstance(new_ob, PathPoint)
    assert isinstance(old_ob, PathPoint)
    delta_g = new_ob.v_gradient - old_ob.v_gradient
    delta_v = new_ob.vspace - old_ob.vspace
    delta_b = new_ob.b_matrix - old_ob.b_matrix
    part1 = np.dot(
        np.dot(new_ob.b_matrix.T, delta_v), new_ob.v_gradient
    )  # v_space gradient here is cartesian
    part2 = np.dot(delta_b.T, new_ob.q_gradient)  # gradient here is internal
    inv_trans = np.dot(new_ob.vspace.T, pse_inv(new_ob.b_matrix.T))
    result = delta_g - np.dot(inv_trans, (part1 + part2))
    return result


def secant_1(new_ob, old_ob):
    """Compute secant value with 1st type.

    Parameters
    ----------
    new_ob : PathPoint
        Previous step point
    old_ob : PathPoint
        Newly computed step point

    Returns
    -------
    np.ndarray(3N)
        computed secont condition value
    """
    assert isinstance(new_ob, PathPoint)
    assert isinstance(old_ob, PathPoint)
    delta_g = new_ob.v_gradient - old_ob.v_gradient
    delta_v = new_ob.vspace - old_ob.vspace
    delta_b = new_ob.b_matrix - old_ob.b_matrix
    part1 = np.dot(
        np.dot(old_ob.b_matrix.T, delta_v), old_ob.v_gradient
    )  # v_space gradient here is cartesian
    part2 = np.dot(delta_b.T, old_ob.q_gradient)  # gradient here is internal
    inv_trans = np.dot(old_ob.vspace.T, pse_inv(old_ob.b_matrix.T))
    result = delta_g - np.dot(inv_trans, (part1 + part2))
    return result


def secant_2(new_ob, old_ob):
    """Compute secant value with 2nd type.

    Parameters
    ----------
    new_ob : PathPoint
        Previous step point
    old_ob : PathPoint
        Newly computed step point

    Returns
    -------
    np.ndarray(3N)
        computed secont condition value
    """
    assert isinstance(new_ob, PathPoint)
    assert isinstance(old_ob, PathPoint)
    delta_g = new_ob.v_gradient - old_ob.v_gradient
    delta_v = new_ob.vspace - old_ob.vspace
    delta_inv_b = pse_inv(new_ob.b_matrix) - pse_inv(old_ob.b_matrix)
    part1 = np.dot(
        np.dot(old_ob.vspace.T, delta_inv_b.T), old_ob.x_gradient
    )  # v_space gradient here is cartesian
    part2 = np.dot(delta_v.T, old_ob.q_gradient)  # gradient here is internal
    return delta_g + part1 + part2


def secant_3(new_ob, old_ob):
    """Compute secant value with 3rd type.

    Parameters
    ----------
    new_ob : PathPoint
        Previous step point
    old_ob : PathPoint
        Newly computed step point

    Returns
    -------
    np.ndarray(3N)
        computed secont condition value
    """
    assert isinstance(new_ob, PathPoint)
    assert isinstance(old_ob, PathPoint)
    delta_g = new_ob.v_gradient - old_ob.v_gradient
    delta_v = new_ob.vspace - old_ob.vspace
    delta_inv_b = pse_inv(new_ob.b_matrix) - pse_inv(old_ob.b_matrix)
    part1 = np.dot(np.dot(old_ob.vspace.T, delta_inv_b.T), old_ob.x_gradient)
    part2 = np.dot(np.dot(new_ob.vspace.T, delta_inv_b.T), new_ob.x_gradient)
    part3 = np.dot(delta_v.T, (old_ob.q_gradient + new_ob.q_gradient))
    return delta_g + 0.5 * (part1 + part2) + 0.5 * part3
