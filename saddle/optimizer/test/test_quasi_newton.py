import numpy as np
import unittest

from saddle.optimizer.quasi_newton import bfgs, psb, sr1, bofill


class TestInternal(unittest.TestCase):

    # def setUp(self):
    #     self.sample_fcn = lambda x, y: x**2 + x*y + 2*y**2
    #     self.sample_g = lambda x, y: np.array([2*x + y, x + 4*y])
    #     self.sample_h = lambda x, y: np.array([[2,1], [1,4]])

    def _set_quadratic(self):
        self.sample_fcn = lambda x, y: x**2 + x * y + 2 * y**2
        self.sample_g = lambda x, y: np.array([2 * x + y, x + 4 * y])
        self.sample_h = lambda x, y: np.array([[2, 1], [1, 4]])
        start_point = np.array([2, 1])
        f_v = self.sample_fcn(*start_point)  # f_v = 8
        self.init_step = np.array([-0.2, -0.2])
        init_g = self.sample_g(*start_point)
        self.init_h = self.sample_h(*start_point)
        new_point = start_point + self.init_step
        new_g = self.sample_g(*new_point)
        self.y = new_g - init_g

    def _set_cubic(self):
        self.sample_fcn = lambda x, y: 1 / 3 * x**3 + x * y + 2 / 3 * y**3
        self.sample_g = lambda x, y: np.array([x**2 + y, x + 2 * y**2])
        self.sample_h = lambda x, y: np.array([[2 * x, 1], [1, 4 * y]])
        start_point = np.array([2, 1])
        f_v = self.sample_fcn(*start_point)  # f_v = 8
        self.init_step = np.array([-0.2, -0.2])
        init_g = self.sample_g(*start_point)
        self.init_h = self.sample_h(*start_point)
        new_point = start_point + self.init_step
        new_g = self.sample_g(*new_point)
        self.y = new_g - init_g
        # assert np.allclose(self.y, np.array([-0.96, -0.92]))

    def test_sr1_quad(self):
        self._set_quadratic()
        new_hessian = sr1(self.init_h, sec_y=self.y, step=self.init_step)
        assert np.allclose(new_hessian, self.init_h)
        assert np.allclose(new_hessian, np.array([[2, 1], [1, 4]]))

    def test_sr1_cubic(self):
        self._set_cubic()
        new_hessian = sr1(self.init_h, sec_y=self.y, step=self.init_step)
        ref_hessian = np.array([[3.93333333, 0.86666667],
                                [0.86666667, 3.73333333]])
        assert np.allclose(new_hessian, ref_hessian)

    def test_psb_quad(self):
        self._set_quadratic()
        new_hessian = psb(self.init_h, sec_y=self.y, step=self.init_step)
        assert np.allclose(new_hessian, self.init_h)
        assert np.allclose(new_hessian, np.array([[2, 1], [1, 4]]))

    def test_psb_cubic(self):
        self._set_cubic()
        new_hessian = psb(self.init_h, sec_y=self.y, step=self.init_step)
        ref_hessian = np.array([[3.95, 0.85], [0.85, 3.75]])
        assert np.allclose(new_hessian, ref_hessian)

    def test_bfgs_quad(self):
        self._set_quadratic()
        new_hessian = bfgs(self.init_h, sec_y=self.y, step=self.init_step)
        assert np.allclose(new_hessian, self.init_h)
        assert np.allclose(new_hessian, np.array([[2, 1], [1, 4]]))

    def test_bfgs_cubic(self):
        self._set_cubic()
        new_hessian = bfgs(self.init_h, sec_y=self.y, step=self.init_step)
        ref_hessian = np.array([[3.95106383, 0.84893617],
                                [0.84893617, 3.75106383]])
        assert np.allclose(new_hessian, ref_hessian)

    # def test_bfgs_cubic_posi_defi(self):
    #     self.sample_fcn = lambda x, y: 1 / 3 * x**3 + x * y + 2 / 3 * y**3
    #     self.sample_g = lambda x, y: np.array([x**2 + y, x + 2 * y**2])
    #     self.sample_h = lambda x, y: np.array([[2 * x, 1], [1, 4 * y]])
    #     start_point = np.array([-3, -2])
    #     f_v = self.sample_fcn(*start_point)  # f_v = 8
    #     init_g = self.sample_g(*start_point)
    #     init_h = self.sample_h(*start_point)
    #     w, v = np.linalg.eigh(init_h)
    #     new_w = np.array([0.2, 0.05])
    #     new_h = np.dot(np.dot(v.T, np.diag(new_w)), v)
    #     self.init_step = np.array([-0.2, -0.2])
    #     new_point = start_point + self.init_step
    #     new_g = self.sample_g(*new_point)
    #     self.y = new_g - init_g
    #     new_hessian = bfgs(new_h, sec_y=self.y, step=self.init_step)
    #     print(np.linalg.eigh(new_hessian))
    #     assert False

    def test_bofill_quad(self):
        self._set_quadratic()
        new_hessian = bofill(self.init_h, sec_y=self.y, step=self.init_step)
        assert np.allclose(new_hessian, self.init_h)
        assert np.allclose(new_hessian, np.array([[2, 1], [1, 4]]))

    def test_bofill_cubic(self):
        self._set_cubic()
        new_hessian = bofill(self.init_h, sec_y=self.y, step=self.init_step)
        ref_hessian_1 = np.array([[3.93333333, 0.86666667],
                                  [0.86666667, 3.73333333]])
        ref_hessian_2 = np.array([[3.95, 0.85], [0.85, 3.75]])
        ref_hessian = 0.9 * ref_hessian_1 + 0.1 * ref_hessian_2
        assert np.allclose(new_hessian, ref_hessian)
