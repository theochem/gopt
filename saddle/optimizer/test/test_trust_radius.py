import numpy as np

from unittest import TestCase
from saddle.optimizer.trust_radius import trust_region_image_potential

class TestTrustRadius(TestCase):

    def setUp(self):
        np.random.seed(199)
        rng_mt = np.random.rand(5, 5)
        self.hessian = np.dot(rng_mt, rng_mt.T)
        self.gradient = np.random.rand(5)

    def test_trim(self):
        step = trust_region_image_potential(self.hessian, self.gradient, 0.02, negative=0)
        assert np.linalg.norm(step) - 0.02 < 1e-7
        step = trust_region_image_potential(self.hessian, self.gradient, 0.1, negative=0)
        assert np.linalg.norm(step) - 1 < 1e-7
        step = trust_region_image_potential(self.hessian, self.gradient, 683, negative=0)
        assert np.linalg.norm(step) - 682.95203408 < 1e-7

    def test_custom_trim(self):
        hessian = np.diag([-1, 3, 5, 7, 9])
        gradient = np.arange(5)
        step = trust_region_image_potential(hessian, gradient, 0.76688102108748257, negative=1)
        ref_step = np.array([-0., -0.3030303, -0.37735849, -0.4109589, -0.43010753])
        assert np.allclose(step, ref_step)
