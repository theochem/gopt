import numpy as np

norm = np.linalg.norm

def energy_based_update(o_gradient, o_hessian, step, diff_energy, *_, min_s, max_s):
    delta_m = np.dot(o_gradient, step) + 0.5 * np.dot(step, np.dot(o_hessian, step))
    ratio = delta_m / diff_energy
    step_size = norm(step)
    if 2/3 < ratio / delta_m < 3/2:
        new_step_size = 2 * step_size
        return np.min(np.max(new_step_size, min_s), max_s)
    elif 1/3 < ratio / delta_m < 3:
        return np.max(step_size, max_s)
    else:
        return np.min(1/4 * step_size, min_s)

def gradient_based_update(o_gradient, o_hessian, step, diff_energy):
    g_predict = o_gradient + np.dot(o_hessian, step)
    rho = (norm(g_predict) - norm(o_gradient)) / (norm())
