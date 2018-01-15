import numpy as np

norm = np.linalg.norm


def energy_based_update(o_gradient, o_hessian, step, diff_energy, *_, min_s,
                        max_s):
    delta_m = np.dot(o_gradient,
                     step) + 0.5 * np.dot(step, np.dot(o_hessian, step))
    ratio = delta_m / diff_energy
    step_size = norm(step)
    if 2 / 3 < ratio / delta_m < 3 / 2:
        new_step_size = 2 * step_size
        return np.min(np.max(new_step_size, min_s), max_s)
    if 1 / 3 < ratio / delta_m < 3:
        return np.max(step_size, max_s)
    return np.min(1 / 4 * step_size, min_s)


def gradient_based_update(o_gradient, o_hessian, n_gradient, step, dof, *_,
                          min_s, max_s):
    step_size = np.linalg.norm(step)
    g_predict = o_gradient + np.dot(o_hessian, step)
    rho = (norm(g_predict) - norm(o_gradient)) / (
        norm(n_gradient) - norm(o_gradient))
    diff_pred = g_predict - o_gradient
    diff_act = n_gradient - o_gradient
    cosine = np.dot(diff_pred, diff_act) / np.dot(
        norm(diff_pred), norm(diff_act))
    p10 = np.sqrt(1.6424 / dof + 1.11 / (dof**2))
    p40 = np.sqrt(0.064175 / dof + 0.0946 / (dof**2))
    if 4 / 5 < rho < 5 / 4 and p10 < cosine:
        new_step = 2 * step_size
        return np.min(np.max(new_step, min_s), max_s)
    if 1 / 5 < rho < 6 and p40 < cosine:
        return np.max(step_size, min_s)
    return min(1 / 2 * step_size, min_s)
