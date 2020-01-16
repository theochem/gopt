from unittest import TestCase

import numpy as np
from importlib_resources import path
from saddle.errors import NotSetError
from saddle.optimizer.path_point import PathPoint
from saddle.reduced_internal import ReducedInternal
from saddle.utils import Utils


class TestPathPoint(TestCase):
    def setUp(self):
        with path("saddle.optimizer.test.data", "water.xyz") as mol_path:
            mol = Utils.load_file(mol_path)
        red_int = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        red_int.add_bond(1, 0)
        red_int.add_bond(1, 2)
        red_int.add_bond(0, 2)
        red_int.add_angle(0, 1, 2)
        red_int.add_angle(1, 0, 2)
        red_int.set_key_ic_number(2)
        self.ri = red_int
        self.pp = PathPoint(red_int)
        self.ri._energy = 5.0
        np.random.seed(10)
        self.ri._energy_gradient = np.random.rand(9)
        with self.assertRaises(NotSetError):
            self.ri.energy_hessian
        assert np.allclose(
            self.pp.q_gradient,
            np.dot(np.linalg.pinv(self.ri.b_matrix.T), self.pp.x_gradient),
        )
        with self.assertRaises(NotSetError):
            self.pp.v_hessian
        self.pp._mod_hessian = np.eye(3)

    def test_basic_property(self):
        assert np.allclose(self.pp.v_hessian, np.eye(3))
        step = -np.dot(np.linalg.pinv(self.pp.v_hessian), self.pp.v_gradient)
        assert np.allclose(self.pp.v_gradient, -step)
        with self.assertRaises(NotSetError):
            self.pp.step = step
        with self.assertRaises(NotSetError):
            self.pp.raw_hessian

    def test_copy_ob_property(self):
        step = -np.dot(np.linalg.pinv(self.pp.v_hessian), self.pp.v_gradient)
        new_pp = self.pp.copy()
        assert new_pp is not self.pp
        new_pp.update_coordinates_with_delta_v(step)
        assert new_pp._step is None
        assert new_pp._stepsize is None
        assert new_pp._mod_hessian is None
        with self.assertRaises(NotSetError):
            new_pp.energy
        with self.assertRaises(NotSetError):
            new_pp.v_gradient
        assert not np.allclose(
            new_pp._instance.coordinates, self.pp._instance.coordinates
        )

    def test_finite_different(self):
        def fct(x, y, z):
            return x ** 2 + y ** 2 + z ** 2

        def grad(x, y, z):
            return np.array([2 * x, 2 * y, 2 * z])

        # hessian = lambda x, y, z: np.eye(3) * 2
        p1, p2 = self._set_point(fct, grad)

        result = PathPoint._calculate_finite_diff_h(p1, p2, 0.001)
        assert np.allclose(result, [2, 0, 0])

        def fct(x, y, z):
            return x ** 3 + 2 * y ** 3 + 3 * z ** 3

        def grad(x, y, z):
            return np.array([3 * x ** 2, 6 * y ** 2, 9 * z ** 2])

        p1, p2 = self._set_point(fct, grad)
        result = PathPoint._calculate_finite_diff_h(p1, p2, 0.001)
        assert np.allclose(result, [6.003, 0, 0])

    def _set_point(self, fct, grad, point=(1, 2, 3), step=(0.001, 0, 0)):
        class T(ReducedInternal):
            def __init__(self):
                pass

            _numbers = [None] * 3
            _ic = [None] * 4

        point_a = T()
        point_a._cc_to_ic_gradient = np.eye(3)
        point_a._vspace = np.eye(3)
        point_b = T()
        point_b._cc_to_ic_gradient = np.eye(3)
        point_b._vspace = np.eye(3)

        new_point = np.array(point) + np.array(step)
        # set init point
        point_a._energy = fct(*point)
        point_a._energy_gradient = grad(*point)
        point_a._internal_gradient = point_a._energy_gradient
        point_a._vspace_gradient = point_a._energy_gradient
        point_a._red_space = True
        point_a._non_red_space = True

        # set new poitn
        point_b._energy = fct(*new_point)
        point_b._energy_gradient = grad(*new_point)
        point_b._internal_gradient = point_b._energy_gradient
        point_b._vspace_gradient = point_b._energy_gradient
        point_b._red_space = True
        point_b._non_red_space = True
        return point_a, point_b

    def test_finite_diff_with_water(self):
        with path("saddle.optimizer.test.data", "water.xyz") as mol_path:
            mol = Utils.load_file(mol_path)
        red_int = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        red_int.auto_select_ic()
        with path("saddle.optimizer.test.data", "water_old.fchk") as fchk_file:
            red_int.energy_from_fchk(fchk_file)
        assert red_int.energy - 75.99264142 < 1e-6
        wt_p1 = PathPoint(red_int=red_int)
        step = [0.001, 0, 0]
        print(wt_p1._instance.vspace)
        ref_vspace = np.array(
            [
                [0.25801783, -0.66522226, 0.70064694],
                [-0.49526649, -0.71373819, -0.49526649],
                [-0.82954078, 0.21921937, 0.51361947],
            ]
        )
        # incase different vspace basis error
        wt_p1._instance.set_vspace(ref_vspace)
        wt_p2 = wt_p1.copy()
        wt_p2.update_coordinates_with_delta_v(step)
        # wt_p2._instance.create_gauss_input(title='water_new')
        with path("saddle.optimizer.test.data", "water_new.fchk") as fchk_file_new:
            wt_p2._instance.energy_from_fchk(fchk_file_new)
        wt_p2._instance.align_vspace(wt_p1._instance)
        assert np.allclose(wt_p1.vspace, wt_p2.vspace)
        result = PathPoint._calculate_finite_diff_h(wt_p1, wt_p2, 0.001)
        assert np.allclose(result, wt_p1._instance.v_hessian[:, 0], atol=1e-2)

    def test_finite_diff_with_water_2(self):
        with path("saddle.optimizer.test.data", "water.xyz") as mol_path:
            mol = Utils.load_file(mol_path)
        red_int = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        red_int.auto_select_ic()
        with path("saddle.optimizer.test.data", "water_old.fchk") as fchk_file:
            red_int.energy_from_fchk(fchk_file)
        assert red_int.energy - 75.99264142 < 1e-6
        red_int.select_key_ic(0)
        ref_v = np.array(
            [
                [-1.00000000e00, -4.17292908e-16, 0.00000000e00],
                [2.10951257e-16, -4.69422035e-01, -8.82973926e-01],
                [3.39185671e-16, -8.82973926e-01, 4.69422035e-01],
            ]
        )
        ref_v2 = np.dot(ref_v, ref_v.T)
        assert np.allclose(ref_v2, np.dot(red_int.vspace, red_int.vspace.T))
        red_int.set_vspace(ref_v)
        wt_p1 = PathPoint(red_int=red_int)
        step = [-0.001, 0, 0]
        wt_p2 = wt_p1.copy()
        wt_p2.update_coordinates_with_delta_v(step)
        # fchk file is for -0.001
        with path("saddle.optimizer.test.data", "water_new_2.fchk") as fchk_file_new:
            wt_p2._instance.energy_from_fchk(fchk_file_new)

        wt_p2._instance.align_vspace(wt_p1._instance)
        assert np.allclose(wt_p1.vspace, wt_p2.vspace)
        result = PathPoint._calculate_finite_diff_h(wt_p1, wt_p2, -0.001)
        assert np.allclose(result, wt_p1._instance.v_hessian[:, 0], atol=1e-2)

    def test_finite_different_with_water_3(self):
        with path("saddle.optimizer.test.data", "water.xyz") as mol_path:
            mol = Utils.load_file(mol_path)
        red_int = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        red_int.auto_select_ic()
        red_int.add_bond(0, 2)
        with path("saddle.optimizer.test.data", "water_old.fchk") as fchk_file:
            red_int.energy_from_fchk(fchk_file)
        assert red_int.energy - 75.99264142 < 1e-6
        red_int.select_key_ic(0)
        wt_p1 = PathPoint(red_int=red_int)
        step = [0.001, 0, 0]
        wt_p2 = wt_p1.copy()
        wt_p2.update_coordinates_with_delta_v(step)
        with path("saddle.optimizer.test.data", "water_new_3.fchk") as fchk_file_new:
            wt_p2._instance.energy_from_fchk(fchk_file_new)
        wt_p2._instance.align_vspace(wt_p1._instance)
        assert np.allclose(wt_p1.vspace, wt_p2.vspace, atol=1e-2)
        result = PathPoint._calculate_finite_diff_h(wt_p1, wt_p2, 0.001)
        assert np.allclose(result, wt_p1._instance.v_hessian[:, 0], atol=1e-2)
