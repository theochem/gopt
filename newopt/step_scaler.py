from __future__ import absolute_import, print_function

import numpy as np

from saddle.solver import ridders_solver


class TRIM(object):  # need tests
    @staticmethod
    def calculate_step(point, negative_eigen=0):
        new_step = TRIM._calculate_step(point.hessian, point.gradient,
                                        point.trust_radius_stride,
                                        negative_eigen)
        point.set_step(new_step)

    @staticmethod
    def _calculate_step(hessian, gradient, trust_radius_stride,
                        nagetive_eigen):
        c_step = -np.dot(np.linalg.pinv(hessian), gradient)
        if np.linalg.norm(c_step) <= trust_radius_stride:
            return c_step
        w, v = np.linalg.eigh(hessian)
        assert np.sum(w < 0.) == nagetive_eigen
        max_w = np.max(w)

        def func_step(value):
            x = w.copy()
            x[:negative] = x[:negative] - value
            x[negative:] = x[negative:] + value
            new_hessian_inv = np.dot(v, np.dot(np.diag(1. / x), v.T))
            return -np.dot(new_hessian_inv, point.gradient)

        def func_value(value):
            step = func_step(value)
            return np.linalg.norm(step) - trust_radius_stride

        while func_value(max_w) >= 0:
            max_w *= 2
        result = ridders_solver(func_value, 0, max_w)
        step = func_step(result)
        return step
