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

    def test_basic_property(self):
        self.ri._energy = 5.
        np.random.seed(10)
        self.ri._energy_gradient = np.random.rand(9)
        self.ri._energy_hessian_transformation()
        with assert_raises(NotSetError):
            self.ri.energy_hessian
        assert np.allclose(self.pp.q_gradient,
            np.dot(np.linalg.pinv(self.ri.b_matrix.T), self.pp.x_gradient))
        with assert_raises(NotSetError):
            self.pp.v_hessian
        self.pp._mod_hessian = np.eye(3)
        assert np.allclose(self.pp.v_hessian, np.eye(3))
        step = -np.dot(np.linalg.pinv(self.pp.v_hessian), self.pp.v_gradient)
        assert np.allclose(self.pp.v_gradient, -step)
