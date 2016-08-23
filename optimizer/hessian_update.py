import numpy as np


def _sr1_update_method(point, point_old, secant_value):
    part1 = secant_value - np.dot(point_old.v_hessian, point_old.stepsize)
    part2 = point_old.stepsize
    half_result = np.dot(part1, part2)
    numerator = np.dot(half_result.T, half_result)
    denominator = np.linalg.norm(part1) ** 2 * np.linalg.norm(part2) ** 2
    result = numerator / denominator
    if result <= 1E-18:
        return deepcopy(point_old.v_hessian)
    else:
        new_value = point_old.v_hessian + \
            np.dot(part1, part1.T) / np.dot(part1, point_old.stepsize)
        return new_value

def _psb_update_method(point, point_old, secant_value):
    part1 = secant_value - np.dot(point_old.v_hessian, point_old.stepsize)
    part2 = point_old.stepsize
    value1 = point_old.v_hessian
    value2 = (np.dot(part1, part2.T) + np.dot(part2, part1.T)) / \
        np.dot(part2.T, part2)
    value3 = np.dot(part2.T, part1) / (np.dot(part2.T, part2) ** 2)
    value4 = np.dot(part2, part2.T)
    new_value = value1 + value2 - np.dot(value3, value4)
    return new_value

def _bfgs_update_method(point, point_old, secant_value):
    part1 = np.dot(point_old.v_hessian, point_old.stepsize)
    part2 = point_old.stepsize
    value1 = point_old.v_hessian
    value2 = np.dot(secant_value, secant_value.T) / \
        np.dot(secant_value.T, part2)
    value3 = np.dot(part1, part1.T) / np.dot(part2.T, part1)
    new_value = value1 + value2 - value3
    return new_value

def _bofill_update_method(point, point_old, secant_value):
    part1 = secant_value - np.dot(point_old.v_hessian, point_old.stepsize)
    part2 = point_old.stepsize
    norm = np.linalg.norm
    psi = 1 - norm(np.dot(part2, part1)) ** 2 / \
        np.dot(norm(part2) ** 2, norm(part1) ** 2)
    result1 = _sr1_update_method(
        point, point_old, secant_value)
    result2 = _psb_update_method(
        point, point_old, secant_value)
    new_value = (1. - psi) * result1 + psi * result2
    return new_value
