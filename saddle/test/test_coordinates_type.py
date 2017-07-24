import os

from saddle.conf import data_dir
from saddle.coordinate_types import (BendAngle, BendCos, BondLength,
                                     ConventionDihedral, NewDihedralCross,
                                     NewDihedralDot)
from saddle.iodata import IOData


class Test_Coordinates_Types(object):
    @classmethod
    def setup_class(self):
        file_path = os.path.join(data_dir, "methanol.xyz")
        mol = IOData.from_file(file_path)
        self.molecule = mol

    def test_bond_length(self):
        bond = BondLength((0, 1), self.molecule.coordinates[[0, 1], ])
        assert bond.value - 2.0220069632957394 < 1e-8
        bond.set_new_coordinates(self.molecule.coordinates[[1, 2], ])
        assert bond.value - 3.3019199546607476 < 1e-8

    def test_bend_angle(self):
        angle = BendAngle((1, 0, 2), self.molecule.coordinates[[1, 0, 2], ])
        assert angle.value - (1.9106254499450943) < 1e-8
        angle.set_new_coordinates(self.molecule.coordinates[[1, 0, 3], ])
        assert angle.value - (1.910636062481526) < 1e-8

    def test_bend_cos(self):
        angle_cos = BendCos((1, 0, 2), self.molecule.coordinates[[1, 0, 2], ])
        assert angle_cos.value - (-0.3333259923254888) < 1e-8
        angle_cos.set_new_coordinates(self.molecule.coordinates[[1, 0, 3], ])
        assert angle_cos.value - (-0.3333359979295637) < 1e-8

    def test_convention_dihedral(self):
        conv_dihed = ConventionDihedral(
            (2, 0, 1, 5), self.molecule.coordinates[[2, 0, 1, 5], ])
        assert conv_dihed.value - 0.5000093782761452 < 1e-8
        conv_dihed.set_new_coordinates(
            self.molecule.coordinates[[3, 0, 1, 5], ])
        assert conv_dihed.value - 0.5000015188648903 < 1e-8

    def test_new_dihed_dot(self):
        new_dihed_dot = NewDihedralDot(
            (2, 0, 1, 5), self.molecule.coordinates[[2, 0, 1, 5], ])
        assert new_dihed_dot.value - 0.33334848858597832 < 1e-8
        new_dihed_dot.set_new_coordinates(
            self.molecule.coordinates[[3, 0, 1, 5], ])
        assert new_dihed_dot.value - 0.33333649967203649 < 1e-8

    def test_new_dihed_cross(self):
        new_dihed_cross = NewDihedralCross(
            (2, 0, 1, 5), self.molecule.coordinates[[2, 0, 1, 5], ])
        assert new_dihed_cross.value - 0.76979948283180566 < 1e-8
        new_dihed_cross.set_new_coordinates(
            self.molecule.coordinates[[3, 0, 1, 5], ])
        assert new_dihed_cross.value - (-0.76980062801256954) < 1e-8
