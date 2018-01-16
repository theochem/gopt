import numpy as np

from unittest import TestCase
from saddle.optimizer.trust_radius_update import (energy_based_update,
                                                  gradient_based_update)


class test_update_trust_radius(TestCase):
    def setUp(self):
        self.func = lambda x, y: x**3 + 2 * x**2 * y + 6 * x * y + 2 * y**3 + 6 * y + 10
        self.gradient = lambda x, y: np.array([3 * x**2 + 4 * x * y + 6 * y, 2*x**2 + 6 * x + 6 * y **2 + 6])
        self.hessian = lambda x, y: np.array([[6*x + 4*y, 4*x + 6], [4*x+6, 12 * y]])
        init = (1, 2)
        assert self.func(*init) == 55
        assert np.allclose(self.gradient(*init), np.array([23, 38]))
        assert np.allclose(self.hessian(*init), np.array([[14, 10], [10, 24]]))

    def test_step_update(self):
        init = np.array((2, 1))
        o_g = self.gradient(*init)
        o_h = self.hessian(*init)
        step = np.array((-1, 1))
        diff = self.func(*(init + step)) - self.func(*init)
        assert diff == 9
        assert np.allclose(o_g, np.array([26, 32]))
        assert np.allclose(o_h, np.array([[16, 14], [14, 12]]))
        est_diff = np.dot(o_g, step) + 0.5 * np.dot(step, np.dot(o_h, step))
        assert est_diff == 6
        ratio = est_diff / diff
        assert ratio - 0.6666666666 < 1e-7
        stepsize = np.linalg.norm(step)
        new_stepsize = energy_based_update(o_g, o_h, step, diff, min_s=1, max_s=5)
        assert new_stepsize == stepsize  # condition 2

        # assert new_stepsize == 5
    def test_step_update_more(self):
        init = np.array((8, 6))
        o_g = self.gradient(*init)
        o_h = self.hessian(*init)
        step = np.array((-2, -1))
        diff = self.func(*(init + step)) - self.func(*init)
        assert diff == -1000
        assert np.allclose(o_g, np.array([420, 398]))
        assert np.allclose(o_h, np.array([[72, 38], [38, 72]]))
        stepsize = np.linalg.norm(step)
        new_stepsize = energy_based_update(o_g, o_h, step, diff, min_s=2, max_s=5)
        assert new_stepsize == 2 * stepsize

        step = np.array((-6, -6))
        stepsize = np.linalg.norm(step)
        diff = self.func(*(init + step)) - self.func(*init)
        new_stepsize = energy_based_update(o_g, o_h, step, diff, min_s=2, max_s=10)
        assert new_stepsize == stepsize


    def test_gradient_update(self):
        init = np.array((8, 6))
        o_g = self.gradient(*init)
        o_h = self.hessian(*init)
        step = np.array((-2, -1))
        stepsize = np.linalg.norm(step)
        n_g = self.gradient(*(init + step))
        diff = n_g - o_g
        assert np.allclose(o_g, np.array([420, 398]))
        assert np.allclose(n_g, np.array([258, 264]))
        assert np.allclose(diff, [-162, -134])
        pre_g = o_g + np.dot(o_h, step)
        assert np.allclose(pre_g, [238, 250])
        new_stepsize = gradient_based_update(o_g, o_h, n_g, step, dof=3, min_s=1, max_s=5)
        assert new_stepsize == 2 * stepsize
