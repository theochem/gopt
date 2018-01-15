import numpy as np

from unittest import TestCase


class test_update_trust_radius(TestCase):

    def setUp(self):
        self.func = lambda x, y: x**3 + 2 * x**2 * y + 6 * x * y + 2 * y**3 + 6 * y + 10
        self.gradient = lambda x, y: np.array([3 * x**2 + 4 * x * y + 6 * y, 2*x**2 + 6 * x + 6 * y **2])
        self.hessian = lambda x, y: np.array([[6*x + 4*y, 4*x + 6], [4*x+6, 12 * y]])
        init = (1, 2)
        assert self.func(*init) == 55
        assert np.allclose(self.gradient(*init), np.array([23, 32]))
        assert np.allclose(self.hessian(*init), np.array([[14, 10],[10, 24]]))
        self.min_s = 0.1
        self.max_s = 1.0

    def test_step_update(self):
        init = np.array((2, 1))
        o_g = self.gradient(*init)
        o_h = self.hessian(*init)
        step = np.array((-3, -2))
        diff = self.func(*(init + step)) - self.func(*init)
        assert diff == -41
        assert np.allclose(o_g, np.array([26, 26]))
        assert np.allclose(o_h, np.array([[16, 14],[14, 12]]))
