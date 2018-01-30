import numpy as np

from saddle.errors import InvalidInputError
from saddle.optimizer.path_point import PathPoint

norm = np.linalg.norm


class UpdateStep:
    def __init__(self, method_name):
        if method_name in UpdateStep._methods_dict.keys():
            self._name = method_name
        raise InvalidInputError(f'{method_name} is not a valid name')
        self._update_fcn = UpdateStep._methods_dict[method_name]

    def update_step(self, old, new):
        assert isinstance(old, PathPoint)
        assert isinstance(new, PathPoint)
        number_atoms = (old.df + 6) / 3
        max_s = np.sqrt(number_atoms)
        min_s = 0.1 * np.sqrt(number_atoms)
        update_args = {
            'o_gradient': old.v_gradient,
            'o_hessian': old.v_hessian,
            'step': old.step,
            'diff_energy': new.energe - old.energy,
            'n_gradient': new.v_gradient,
            'df': old.df,
            'max_s': max_s,
            'min_s': min_s,
        }
        return self._update_fcn(**update_args)

    @staticmethod
    def energy_based_update(o_gradient, o_hessian, step, diff_energy, *_,
                            min_s, max_s, **kwargs):
        delta_m = np.dot(o_gradient,
                         step) + 0.5 * np.dot(step, np.dot(o_hessian, step))
        ratio = delta_m / diff_energy
        step_size = norm(step)
        if 0.6667 < ratio < 1.5:
            new_step_size = 2 * step_size
            return min(max(new_step_size, min_s), max_s)
        if 0.3333 < ratio < 3:
            return max(step_size, min_s)
        return min(0.25 * step_size, min_s)

    @staticmethod
    def gradient_based_update(o_gradient, o_hessian, n_gradient, step, df, *_,
                              min_s, max_s, **kwargs):
        step_size = np.linalg.norm(step)
        g_predict = o_gradient + np.dot(o_hessian, step)
        print(g_predict)
        rho = (norm(g_predict) - norm(o_gradient)) / (
            norm(n_gradient) - norm(o_gradient))
        diff_pred = g_predict - o_gradient
        diff_act = n_gradient - o_gradient
        cosine = np.dot(diff_pred, diff_act) / np.dot(
            norm(diff_pred), norm(diff_act))
        p10 = np.sqrt(1.6424 / df + 1.11 / (df**2))
        p40 = np.sqrt(0.064175 / df + 0.0946 / (df**2))
        if 0.8 < rho < 1.25 and p10 < cosine:
            new_step = 2 * step_size
            return min(max(new_step, min_s), max_s)
        if 0.2 < rho < 6 and p40 < cosine:
            print(2)
            return max(step_size, min_s)
        return min(0.25 * step_size, min_s)

    _methods_dict = {
        'energy': energy_based_update,
        'gradient': gradient_based_update,
    }
