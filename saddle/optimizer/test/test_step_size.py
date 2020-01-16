import numpy as np

from unittest import TestCase
from saddle.optimizer.step_size import Stepsize
from saddle.optimizer.path_point import PathPoint

# function alias
energy_based_update = Stepsize.energy_based_update
gradient_based_update = Stepsize.gradient_based_update


class test_update_trust_radius(TestCase):
    def setUp(self):
        "set up test function gradient and hessian"
        self.func = (
            lambda x, y: x ** 3 + 2 * x ** 2 * y + 6 * x * y + 2 * y ** 3 + 6 * y + 10
        )
        self.gradient = lambda x, y: np.array(
            [3 * x ** 2 + 4 * x * y + 6 * y, 2 * x ** 2 + 6 * x + 6 * y ** 2 + 6]
        )
        self.hessian = lambda x, y: np.array(
            [[6 * x + 4 * y, 4 * x + 6], [4 * x + 6, 12 * y]]
        )
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
        new_stepsize = energy_based_update(
            o_g, o_h, step, diff, stepsize, min_s=1, max_s=5
        )
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
        new_stepsize = energy_based_update(
            o_g, o_h, step, diff, stepsize, min_s=2, max_s=5
        )
        assert new_stepsize == 2 * stepsize

        step = np.array((-6, -6))
        stepsize = np.linalg.norm(step)
        diff = self.func(*(init + step)) - self.func(*init)
        new_stepsize = energy_based_update(
            o_g, o_h, step, diff, stepsize, min_s=2, max_s=10
        )
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
        assert np.allclose(o_h, np.array([[72, 38], [38, 72]]))
        assert np.allclose(diff, [-162, -134])
        pre_g = o_g + np.dot(o_h, step)
        assert np.allclose(pre_g, [238, 250])
        new_stepsize = gradient_based_update(
            o_g, o_h, n_g, step, df=3, step_size=stepsize, min_s=1, max_s=5
        )
        assert new_stepsize == 2 * stepsize

        step = list(map(int, (-np.dot(np.linalg.pinv(o_h), o_g))))
        stepsize = np.linalg.norm(step)
        assert np.allclose(step, [-4, -3])
        assert stepsize == 5  # step == [-4, -3]
        n_g = self.gradient(*(init + step))
        assert np.allclose(n_g, [114, 116])
        diff = n_g - o_g
        assert np.allclose(diff, [-306, -282])
        pre_g = o_g + np.dot(o_h, step)
        assert np.allclose(pre_g, [18, 30])
        new_stepsize = gradient_based_update(
            o_g, o_h, n_g, step, df=3, step_size=stepsize, min_s=1, max_s=5
        )
        assert new_stepsize == 5

    def _set_path_points(self):
        "create a class for testing points"

        class Other(PathPoint):
            def __init__(self):
                pass

            @property
            def df(self):
                return 1

        class Attr:
            pass

        self.p1 = Other()
        self.p2 = Other()
        start_point = np.array([2, 1])
        self.p1._instance = Attr()
        self.p2._instance = Attr()

        self.p1._instance.energy = self.func(*start_point)
        self.p1._instance.b_matrix = np.eye(2)
        self.p1._instance.vspace = np.eye(2)
        self.p1._instance.v_gradient = self.gradient(*start_point)
        self.p1._instance.q_gradient = self.p1._instance.v_gradient
        self.p1._instance.x_gradient = self.p1._instance.v_gradient
        self.p1._instance._df = self.p1
        self.p1._mod_hessian = self.hessian(*start_point)
        self.p1._step = np.array([-1, 1])

        new_point = np.array([1, 2])
        self.p2._instance.energy = self.func(*new_point)
        self.p2._instance.b_matrix = np.eye(2)
        self.p2._instance.vspace = np.eye(2)
        self.p2._instance.v_gradient = self.gradient(*new_point)
        self.p2._instance.q_gradient = self.p2._instance.v_gradient
        self.p2._instance.x_gradient = self.p2._instance.v_gradient

    def test_update_object(self):
        self._set_path_points()
        with self.assertRaises(ValueError):
            energy_ob = Stepsize("gibberish")
        assert np.allclose(self.p1.v_gradient, [26, 32])
        assert np.allclose(self.p2.v_gradient, [23, 38])
        energy_ob = Stepsize("energy")
        new_step = energy_ob.update_step(old=self.p1, new=self.p2)
        assert np.allclose(new_step, self.p1.stepsize)
        gradient_ob = Stepsize("gradient")
        new_step = gradient_ob.update_step(old=self.p1, new=self.p2)
        assert np.allclose(new_step, energy_ob.min_s)
