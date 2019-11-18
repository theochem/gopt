import unittest

import numpy as np
from importlib_resources import path
from numpy.testing import assert_allclose
from saddle.optimizer.path_point import PathPoint
from saddle.optimizer.secant import secant, secant_1, secant_2, secant_3
from saddle.reduced_internal import ReducedInternal
from saddle.utils import Utils


class TestSecant(unittest.TestCase):
    def setUp(self):
        with path('saddle.test.data', 'water.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        self.old_ob = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        self.old_ob.add_bond(0, 1)
        self.old_ob.add_bond(1, 2)
        self.old_ob.add_angle(0, 1, 2)
        with path('saddle.optimizer.test.data', 'water_old.fchk') as fchk_file1:
            self.old_ob.energy_from_fchk(fchk_file1)
        self.new_ob = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        self.new_ob.add_bond(0, 1)
        self.new_ob.add_bond(1, 2)
        self.new_ob.add_angle(0, 1, 2)
        with path('saddle.optimizer.test.data', 'water_new.fchk') as fchk_file2:
            self.new_ob.energy_from_fchk(fchk_file2)
        self.new_ob.align_vspace(self.old_ob)
        assert_allclose(self.new_ob.vspace, self.old_ob.vspace, atol=1e-6)
        self.newp = PathPoint(self.new_ob)
        self.oldp = PathPoint(self.old_ob)

    def test_secant_condition(self):
        result = secant(self.newp, self.oldp)

        # separate calculation
        part1 = self.newp.v_gradient - self.oldp.v_gradient
        part2 = np.dot(self.newp.vspace.T,
                       np.linalg.pinv(self.newp.b_matrix.T))
        part3 = np.dot(
            self.newp.b_matrix.T,
            np.dot((self.newp.vspace - self.oldp.vspace),
                   self.newp.v_gradient))
        part4 = np.dot((self.newp.b_matrix - self.oldp.b_matrix).T,
                       self.newp.q_gradient)
        final = part1 - np.dot(part2, (part3 + part4))
        assert np.allclose(result, final, atol=1e-6)

    def test_secant_condition_0(self):
        d_s = np.dot(self.old_ob.vspace.T, self.new_ob.ic_values - self.old_ob.ic_values)
        ref = np.dot(self.old_ob.v_hessian, d_s)
        result = secant(self.newp, self.oldp)
        assert_allclose(ref, result, atol=1e-5)

    def test_secant_condition_1(self):
        d_s = np.dot(self.old_ob.vspace.T, self.new_ob.ic_values - self.old_ob.ic_values)
        ref = np.dot(self.old_ob.v_hessian, d_s)
        result = secant_1(self.newp, self.oldp)
        assert_allclose(ref, result, atol=1e-5)

    def test_secant_condition_2(self):
        d_s = np.dot(self.old_ob.vspace.T, self.new_ob.ic_values - self.old_ob.ic_values)
        ref = np.dot(self.old_ob.v_hessian, d_s)
        result = secant_2(self.newp, self.oldp)
        assert_allclose(ref, result, atol=1e-5)

    def test_secant_condition_3(self):
        d_s = np.dot(self.old_ob.vspace.T, self.new_ob.ic_values - self.old_ob.ic_values)
        ref = np.dot(self.old_ob.v_hessian, d_s)
        result = secant_3(self.newp, self.oldp)
        assert_allclose(ref, result, atol=1e-5)
