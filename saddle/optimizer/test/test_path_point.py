import numpy as np

from unittest import TestCase
from numpy.testing import assert_raises

from pkg_resources import Requirement, resource_filename
from saddle.optimizer.path_point import PathPoint
from saddle.iodata import IOData
from saddle.reduced_internal import ReducedInternal
from saddle.errors import NotSetError


class TestPathPoint(TestCase):
    def setUp(self):
        mol_path = resource_filename(
            Requirement.parse('saddle'), 'data/water.xyz')
        mol = IOData.from_file(mol_path)
        red_int = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        red_int.add_bond(1, 0)
        red_int.add_bond(1, 2)
        red_int.add_bond(0, 2)
        red_int.add_angle_cos(0, 1, 2)
        red_int.add_angle_cos(1, 0, 2)
        red_int.set_key_ic_number(2)
        self.ri = red_int
        self.pp = PathPoint(red_int)
        self.ri._energy = 5.
        np.random.seed(10)
        self.ri._energy_gradient = np.random.rand(9)
        self.ri._energy_hessian_transformation()
        with assert_raises(NotSetError):
            self.ri.energy_hessian
        assert np.allclose(self.pp.q_gradient,
                           np.dot(
                               np.linalg.pinv(self.ri.b_matrix.T),
                               self.pp.x_gradient))
        with assert_raises(NotSetError):
            self.pp.v_hessian
        self.pp._mod_hessian = np.eye(3)

    def test_basic_property(self):
        assert np.allclose(self.pp.v_hessian, np.eye(3))
        step = -np.dot(np.linalg.pinv(self.pp.v_hessian), self.pp.v_gradient)
        assert np.allclose(self.pp.v_gradient, -step)
        with assert_raises(NotSetError):
            self.pp.step = step
        with assert_raises(NotSetError):
            self.pp.raw_hessian

    def test_copy_ob_property(self):
        step = -np.dot(np.linalg.pinv(self.pp.v_hessian), self.pp.v_gradient)
        new_pp = self.pp.copy()
        assert new_pp is not self.pp
        new_pp.update_coordinates_with_delta_v(step)
        assert new_pp._step is None
        assert new_pp._stepsize is None
        assert new_pp._mod_hessian is None
        with assert_raises(NotSetError):
            new_pp.energy
        with assert_raises(NotSetError):
            new_pp.v_gradient
        assert not np.allclose(new_pp._instance.coordinates,
                               self.pp._instance.coordinates)

    def test_finite_different(self):
        fct = lambda x, y, z: x**2 + y**2 + z**2
        grad = lambda x, y, z: np.array([2 * x, 2 * y, 2 * z])
        # hessian = lambda x, y, z: np.eye(3) * 2
        p1, p2 = self._set_point(fct, grad)


        result = PathPoint._calculate_finite_diff_h(p1, p2)
        assert np.allclose(result, [2, 0, 0])


    def _set_point(self, fct, grad):
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

        point = (1, 2, 3)
        new_point = (1.001, 2, 3)
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
