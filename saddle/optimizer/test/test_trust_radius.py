import numpy as np

from unittest import TestCase
from saddle.optimizer.trust_radius import TrustRegion
from saddle.optimizer.path_point import PathPoint

trim = TrustRegion.trim


# pylint: disable=E1101, E1133
# Disable pylint on numpy.random functions
class TestTrustRadius(TestCase):
    def setUp(self):
        np.random.seed(199)
        rng_mt = np.random.rand(5, 5)
        self.hessian = np.dot(rng_mt, rng_mt.T)
        self.gradient = np.random.rand(5)

    def test_trim(self):
        step = trim(self.hessian, self.gradient, 0.02)
        assert np.linalg.norm(step) - 0.02 < 1e-7
        step = trim(self.hessian, self.gradient, 0.1)
        assert np.linalg.norm(step) - 1 < 1e-7
        step = trim(self.hessian, self.gradient, 683)
        assert np.linalg.norm(step) - 682.95203408 < 1e-7

    def test_trim_one_neg(self):
        hessian = np.diag([-1, 3, 5, 7, 9])
        gradient = np.arange(5)
        step = trim(hessian, gradient, 0.766881021)
        ref_step = np.array(
            [-0., -0.3030303, -0.37735849, -0.4109589, -0.43010753])
        assert np.allclose(step, ref_step)

    def test_trim_all_pos(self):
        hessian = np.diag([2, 4, 6, 7, 11])
        gradient = np.arange(1, 6)
        step = trim(hessian, gradient, 0.8451898886)
        ref_step = np.array(
            [-0.27027027, -0.35087719, -0.38961039, -0.45977011, -0.39370079])
        assert np.allclose(step, ref_step)

    def test_two_neg(self):
        hessian = np.diag([-2, -4, 6, 7, 11])
        gradient = np.arange(1, 6)
        step = trim(hessian, gradient, 0.8451898886)
        ref_step = np.array(
            [0.27027027, 0.35087719, -0.38961039, -0.45977011, -0.39370079])
        assert np.allclose(step, ref_step)

    def _quad_func_setup(self):
        # function f = x^2 + 2y^2 + 3xy + 2x + 4y + 1
        self.gradient = lambda x, y: np.array(
            [2 * x + 3 * y + 2, 4 * y + 3 * x + 4])
        self.hessian = lambda x, y: np.array([[2, 3], [3, 4]])

    def _set_path_points(self):
        class Other(PathPoint):
            def __init__(self):
                pass

        class Attr:
            pass

        self.p1 = Other()
        start_point = np.array([2, 1])
        self.p1._instance = Attr()
        self.p1._instance.v_gradient = self.gradient(*start_point)
        self.p1._mod_hessian = self.hessian(*start_point)
        self.p1._stepsize = 1

    def test_ob_trust_raiuds(self):
        self._quad_func_setup()
        self._set_path_points()
        trim_ob = TrustRegion('trim')
        assert np.allclose(self.p1.v_gradient, np.array([9, 14]))
        assert np.allclose(self.p1.v_hessian, np.array([[2, 3], [3, 4]]))
        self.p1.step_hessian = self.p1.v_hessian
        step = trim_ob.calculate_trust_step(self.p1)
        assert np.linalg.norm(step) - 1 < 1e-6
        self.p1._stepsize = 1.8353865
        step = trim_ob.calculate_trust_step(self.p1)
        assert np.allclose(step, np.array([-1.28760195, -1.30794685]))

        self.p1._instance.v_gradient = self.gradient(0, -1)
        self.p1._mod_hessian = self.hessian(0, -1)
        self.p1._stepsize = 6
        step = trim_ob.calculate_trust_step(self.p1)
        assert np.allclose(self.p1.v_gradient, np.array([-1, 0]))
        assert np.linalg.norm(step) - 5 < 1e-5
        self.p1._stepsize = 0.2643558
        step = trim_ob.calculate_trust_step(self.p1)
        assert np.allclose(step, np.array([-0.17079935, 0.20177115]))
