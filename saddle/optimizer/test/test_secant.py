import numpy as np
import unittest

from saddle.reduced_internal import ReducedInternal
from saddle.optimizer.secant import secant
from pkg_resources import Requirement, resource_filename
from saddle.iodata import IOData


class TestSecant(unittest.TestCase):
    def setUp(self):
        mol_path = resource_filename(
            Requirement.parse('saddle'), 'data/water.xyz')
        mol = IOData.from_file(mol_path)
        self.old_ob = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        self.old_ob.add_bond(0, 1)
        self.old_ob.add_bond(1, 2)
        self.old_ob.add_angle_cos(0, 1, 2)
        self.old_ob.add_bond(0, 2)
        self.old_ob.add_angle_cos(0, 2, 1)
        self.new_ob = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        self.new_ob.add_bond(0, 1)
        self.new_ob.add_bond(1, 2)
        self.new_ob.add_angle_cos(0, 1, 2)
        self.new_ob.set_target_ic([1.0, 1.0, -0.5])
        self.new_ob.converge_to_target_ic()
        self.new_ob.add_bond(0, 2)
        self.new_ob.add_angle_cos(0, 2, 1)
        np.random.seed(100)
        self.old_ob._energy_gradient = np.random.rand(9)
        self.old_ob._gradient_transform()
        self.new_ob._energy_gradient = np.random.rand(9)
        self.new_ob._gradient_transform()

    def test_secant_condition(self):
        result = secant(self.new_ob, self.old_ob)

        # separate calculation
        part1 = self.new_ob.vspace_gradient - self.old_ob.vspace_gradient
        part2 = np.dot(self.new_ob.vspace.T,
                       np.linalg.pinv(self.new_ob.b_matrix.T))
        part3 = np.dot(self.new_ob.b_matrix.T,
                       np.dot((self.new_ob.vspace - self.old_ob.vspace),
                              self.new_ob.vspace_gradient))
        part4 = np.dot((self.new_ob.b_matrix - self.old_ob.b_matrix).T,
                       self.new_ob.q_gradient)
        final = part1 - np.dot(part2, (part3 + part4))
        assert np.allclose(result, final)
