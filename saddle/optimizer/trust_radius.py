import numpy as np

from saddle.solver import ridders_solver


class TrustRegion:
    @staticmethod
    def trust_region_image_potential(hessian, gradient, stepsize):
        assert stepsize > 0
        val, vectors = np.linalg.eigh(hessian)
        negative = np.sum([val < 0])

        def value_func(lamd):
            values = val.copy()
            values[:negative] -= lamd
            values[negative:] += lamd
            assert np.all(values != 0)
            n_v = 1. / values
            new_h = np.dot(vectors, np.dot(np.diag(n_v), vectors.T))
            return -np.dot(new_h, gradient)

        def value_compare(lamd):
            step = value_func(lamd)
            return stepsize - np.linalg.norm(step)

        if value_compare(0) >= 0:  # inital case
            return value_func(0)
        start_value = 1  # need to optimized in the future
        if value_compare(start_value) >= 0:  # initial iteration case
            answer = ridders_solver(value_compare, 0, start_value)
            # print(answer)
            return value_func(answer)
        while value_compare(start_value) < 0:
            # print(start_value, value_compare(start_value))
            start_value *= 2
            if value_compare(start_value) >= 0:
                answer = ridders_solver(value_compare, start_value / 2,
                                        start_value)
                # print(answer)
                return value_func(answer)

    trim = trust_region_image_potential

    @staticmethod
    def rational_functional_optimization(hessian, gradient, stepsize):
        raise NotImplementedError

    rfo = rational_functional_optimization

    _trust_radius_methods = {
        "trim": trim.__func__,
        "rfo": rfo.__func__,
    }
