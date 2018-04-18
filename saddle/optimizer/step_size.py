import numpy as np

from saddle.optimizer.path_point import PathPoint

norm = np.linalg.norm


class Stepsize:
    def __init__(self, method_name):
        if method_name not in Stepsize._methods_dict.keys():
            raise ValueError(f'{method_name} is not a valid name')
        self._name = method_name
        self._update_fcn = Stepsize._methods_dict[method_name]
        self._max_s = None
        self._min_s = None
        self._init_s = None
        self._init_flag = False

    @property
    def min_s(self):
        return self._min_s

    @property
    def max_s(self):
        return self.max_s

    def initialize(self, init_point, ratio=0.35):
        assert init_point.df > 0
        number_of_atoms = (init_point.df + 6) // 3
        self._max_s = np.sqrt(number_of_atoms)
        self._min_s = 0.1 * self._max_s
        self._init_s = ratio * self._max_s
        init_point.stepsize = self._init_s
        self._init_flag = True

    def update_step(self, old, new):
        if not isinstance(old, PathPoint) or not isinstance(new, PathPoint):
            raise TypeError("Improper input type for {old} or {new}")
        if self._init_flag is False:
            self.initialize(old)
        update_args = {
            'o_gradient': old.v_gradient,
            'o_hessian': old.v_hessian,
            'step': old.step,
            'diff_energy': new.energy - old.energy,
            'n_gradient': new.v_gradient,
            'df': old.df,
            'max_s': self._max_s,
            'min_s': self._min_s,
            'step_size': old.stepsize
        }
        return self._update_fcn(**update_args)

    @staticmethod
    def energy_based_update(o_gradient, o_hessian, step, diff_energy,
                            step_size, *_, min_s, max_s, **kwargs):
        delta_m = np.dot(o_gradient,
                         step) + 0.5 * np.dot(step, np.dot(o_hessian, step))
        ratio = delta_m / diff_energy
        if 0.6667 < ratio < 1.5:
            new_step_size = 2 * step_size
            return min(max(new_step_size, min_s), max_s)
        if 0.3333 < ratio < 3:
            return max(step_size, min_s)
        return min(0.25 * step_size, min_s)

    @staticmethod
    def gradient_based_update(o_gradient, o_hessian, n_gradient, step, df,
                              step_size, *_, min_s, max_s, **kwargs):
        g_predict = o_gradient + np.dot(o_hessian, step)
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
            return max(step_size, min_s)
        return min(0.5 * step_size, min_s)

    _methods_dict = {
        'energy': energy_based_update.__func__,
        'gradient': gradient_based_update.__func__,
    }
