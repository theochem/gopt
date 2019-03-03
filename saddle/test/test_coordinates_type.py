from unittest import TestCase

import numpy as np
from importlib_resources import path
from saddle.coordinate_types import (BendAngle, BendCos, BondLength,
                                     ConventionDihedral, DihedralAngle,
                                     NewDihedralCross, NewDihedralDot)
from saddle.errors import NotSetError
from saddle.molmod import bend_angle
from saddle.utils import Utils


class Test_Coordinates_Types(TestCase):
    @classmethod
    def setup_class(self):
        with path('saddle.test.data', 'methanol.xyz') as file_path:
            mol = Utils.load_file(file_path)
        self.molecule = mol

        with path('saddle.test.data', 'h2o2.xyz') as file_path2:
            mol2 = Utils.load_file(file_path2)
        self.h2o2 = mol2

    def test_bond_length(self):
        bond = BondLength((0, 1), self.molecule.coordinates[[0, 1]])
        assert bond.value - 2.0220069632957394 < 1e-8
        bond.set_new_coordinates(self.molecule.coordinates[[1, 2]])
        assert bond.value - 3.3019199546607476 < 1e-8
        # set target
        bond.target = 3.0
        # calculate ref
        ref_v = (bond.value - 3.0)**2
        real_v = bond.cost_v
        assert np.allclose(ref_v, bond.cost_v)
        ref_d = (bond.cost_v - (bond.value - 1e-6 - 3)**2) / 1e-6
        real_d = bond.cost_d
        assert np.allclose(ref_d, bond.cost_d, atol=1e-6)
        ref_dd = (bond.cost_d - 2 * (bond.value - 1e-6 - 3)) / 1e-6
        real_dd = bond.cost_dd
        assert np.allclose(ref_dd, bond.cost_dd, atol=1e-6)
        bond.weight = 1.2
        assert np.allclose(real_v * 1.2, bond.cost_v)
        assert np.allclose(real_d * 1.2, bond.cost_d)
        assert np.allclose(real_dd * 1.2, bond.cost_dd)

    def test_bend_angle(self):
        angle = BendAngle((1, 0, 2), self.molecule.coordinates[[1, 0, 2]])
        assert angle.value - (1.9106254499450943) < 1e-8
        angle.set_new_coordinates(self.molecule.coordinates[[1, 0, 3]])
        assert angle.value - (1.910636062481526) < 1e-8
        # calculate ref value
        angle.target = 1.8
        # calculate ref v
        ref_v = (np.cos(angle.value) - np.cos(1.8))**2
        real_v = angle.cost_v
        assert np.allclose(ref_v, angle.cost_v)
        # calculate ref d
        ref_d = (angle.cost_v -
                 (np.cos(angle.value - 1e-6) - np.cos(1.8))**2) / 1e-6
        real_d = angle.cost_d
        # print(ref_d, angle.cost_d)
        assert np.allclose(ref_d, angle.cost_d, atol=1e-6)
        # calculate ref dd
        ref_d2 = -2 * (np.cos(angle.value - 1e-6) -
                       np.cos(1.8)) * np.sin(angle.value - 1e-6)
        ref_dd = (angle.cost_d - ref_d2) / 1e-6
        real_dd = angle.cost_dd
        assert np.allclose(ref_dd, angle.cost_dd, atol=1e-6)
        # test weight
        angle.weight = 1.2
        assert np.allclose(angle.cost_v, real_v * 1.2)
        assert np.allclose(angle.cost_d, real_d * 1.2)
        assert np.allclose(angle.cost_dd, real_dd * 1.2)

    def test_bend_cos(self):
        angle_cos = BendCos((1, 0, 2), self.molecule.coordinates[[1, 0, 2]])
        assert angle_cos.value - (-0.3333259923254888) < 1e-8
        angle_cos.set_new_coordinates(self.molecule.coordinates[[1, 0, 3]])
        assert angle_cos.value - (-0.3333359979295637) < 1e-8

    def test_dihed_angle(self):
        dihed_angle = DihedralAngle((2, 0, 1, 3),
                                    self.h2o2.coordinates[[2, 0, 1, 3]])
        assert dihed_angle.value - 1.43966112870 < 1e-8
        with self.assertRaises(NotSetError):
            dihed_angle.target
        dihed_angle.target = 1.57
        v = dihed_angle.cost_v
        d = dihed_angle.cost_d
        dd = dihed_angle.cost_dd
        # calculate ref value
        sin_ang1 = np.sin(bend_angle(dihed_angle._coordinates[:3]))**2
        sin_ang2 = np.sin(bend_angle(dihed_angle._coordinates[1:]))**2
        ref_v = sin_ang1 * sin_ang2 * (
            2 - 2 * np.cos(dihed_angle.value - dihed_angle.target))
        assert np.allclose(ref_v, v)
        # finite diff for g
        ref_v2 = sin_ang1 * sin_ang2 * (2 - 2 * np.cos(
            (dihed_angle.value - 1e-6) - dihed_angle.target))
        ref_d = (ref_v - ref_v2) / 1e-6
        assert np.allclose(ref_d, d, atol=1e-6)
        # finite diff for h
        ref_d1 = sin_ang1 * sin_ang2 * 2 * np.sin(dihed_angle.value -
                                                  dihed_angle.target)
        assert np.allclose(ref_d1, d)
        ref_d2 = sin_ang1 * sin_ang2 * 2 * np.sin(dihed_angle.value - 1e-6 -
                                                  dihed_angle.target)
        ref_dd = (ref_d1 - ref_d2) / 1e-6
        assert np.allclose(ref_dd, dd, atol=1e-6)
        dihed_angle.weight = 1.2
        assert np.allclose(dd * 1.2, dihed_angle.cost_dd)
        assert np.allclose(d * 1.2, dihed_angle.cost_d)
        assert np.allclose(v * 1.2, dihed_angle.cost_v)

    def test_convention_dihedral(self):
        conv_dihed = ConventionDihedral(
            (2, 0, 1, 5), self.molecule.coordinates[[2, 0, 1, 5]])
        assert conv_dihed.value - 0.5000093782761452 < 1e-8
        conv_dihed.set_new_coordinates(
            self.molecule.coordinates[[3, 0, 1, 5], ])
        assert conv_dihed.value - 0.5000015188648903 < 1e-8

    def test_new_dihed_dot(self):
        new_dihed_dot = NewDihedralDot((2, 0, 1, 5),
                                       self.molecule.coordinates[[2, 0, 1, 5]])
        assert new_dihed_dot.value - 0.33334848858597832 < 1e-8
        new_dihed_dot.set_new_coordinates(
            self.molecule.coordinates[[3, 0, 1, 5]])
        assert new_dihed_dot.value - 0.33333649967203649 < 1e-8

    def test_new_dihed_cross(self):
        new_dihed_cross = NewDihedralCross(
            (2, 0, 1, 5), self.molecule.coordinates[[2, 0, 1, 5]])
        assert new_dihed_cross.value - 0.76979948283180566 < 1e-8
        new_dihed_cross.set_new_coordinates(
            self.molecule.coordinates[[3, 0, 1, 5]])
        assert new_dihed_cross.value - (-0.76980062801256954) < 1e-8
