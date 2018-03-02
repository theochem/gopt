import numpy as np

from unittest import TestCase
from saddle.optimizer.trust_radius import TrustRegion

trim = TrustRegion.trim

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
        ref_step = np.array([-0., -0.3030303, -0.37735849, -0.4109589, -0.43010753])
        assert np.allclose(step, ref_step)

    def test_trim_all_pos(self):
        hessian = np.diag([2, 4, 6, 7, 11])
        gradient = np.arange(1, 6)
        step = trim(hessian, gradient, 0.8451898886)
        ref_step = np.array([-0.27027027, -0.35087719, -0.38961039, -0.45977011, -0.39370079])
        assert np.allclose(step, ref_step)

    def test_two_neg(self):
        hessian = np.diag([-2, -4, 6, 7, 11])
        gradient = np.arange(1, 6)
        step = trim(hessian, gradient, 0.8451898886)
        ref_step = np.array([0.27027027, 0.35087719, -0.38961039, -0.45977011, -0.39370079])
        assert np.allclose(step, ref_step)
