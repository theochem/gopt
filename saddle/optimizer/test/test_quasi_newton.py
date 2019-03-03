import numpy as np

from unittest import TestCase

from saddle.optimizer.quasi_newton import QuasiNT
from saddle.optimizer.path_point import PathPoint
bfgs = QuasiNT.bfgs
psb = QuasiNT.psb
sr1 = QuasiNT.sr1
bofill = QuasiNT.bofill

# , psb, sr1, bofill


class TestInternal(TestCase):

    # def setUp(self):
    #     self.sample_fcn = lambda x, y: x**2 + x*y + 2*y**2
    #     self.sample_g = lambda x, y: np.array([2*x + y, x + 4*y])
    #     self.sample_h = lambda x, y: np.array([[2,1], [1,4]])

    def _set_quadratic(self):
        self.sample_fcn = lambda x, y: x**2 + x * y + 2 * y**2
        self.sample_g = lambda x, y: np.array([2 * x + y, x + 4 * y])
        self.sample_h = lambda x, y: np.array([[2, 1], [1, 4]])
        start_point = np.array([2, 1])
        # f_v = self.sample_fcn(*start_point)  # f_v = 8
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
        # f_v = self.sample_fcn(*start_point)  # f_v = 8
        self.init_step = np.array([-0.2, -0.2])
        init_g = self.sample_g(*start_point)
        self.init_h = self.sample_h(*start_point)
        new_point = start_point + self.init_step
        new_g = self.sample_g(*new_point)
        self.y = new_g - init_g
        # assert np.allclose(self.y, np.array([-0.96, -0.92]))

    def _set_path_points(self):
        class Other(PathPoint):
            def __init__(self):
                pass

        class Attr:
            pass

        self.p1 = Other()
        self.p2 = Other()
        start_point = np.array([2, 1])
        self.p1._instance = Attr()
        self.p2._instance = Attr()
        self.p1._instance.b_matrix = np.eye(2)
        self.p1._instance.vspace = np.eye(2)
        self.p1._instance.v_gradient = self.sample_g(*start_point)
        self.p1._instance.q_gradient = self.p1._instance.v_gradient
        self.p1._instance.x_gradient = self.p1._instance.v_gradient
        self.p1._mod_hessian = self.sample_h(*start_point)
        self.p1._step = np.array([-0.2, -0.2])
        new_point = np.array([1.8, 0.8])
        self.p2._instance.b_matrix = np.eye(2)
        self.p2._instance.vspace = np.eye(2)
        self.p2._instance.v_gradient = self.sample_g(*new_point)
        self.p2._instance.q_gradient = self.p2._instance.v_gradient
        self.p2._instance.x_gradient = self.p2._instance.v_gradient

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

    def test_quasi_object(self):
        self._set_quadratic()
        with self.assertRaises(ValueError):
            qnt = QuasiNT('gibberish')
        qnt = QuasiNT('sr1')
        with self.assertRaises(TypeError):
            qnt.update_hessian(1, 2)
        # setup test points
        self._set_path_points()

        new_hessian = qnt.update_hessian(self.p1, self.p2)
        assert np.allclose(new_hessian, np.array([[2, 1], [1, 4]]))
