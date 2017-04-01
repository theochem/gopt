import os

import numpy as np

from saddle.iodata import IOData
from saddle.coordinate_types import (BendAngle, BondLength, ConventionDihedral,
                                     NewDihedralCross, NewDihedralDot)
from saddle.internal import Internal

from copy import deepcopy


class Test_Coordinates_Types(object):
    @classmethod
    def setup_class(self):
        path = os.path.dirname(os.path.realpath(__file__))
        mol = IOData.from_file(path + "/../data/methanol.xyz")
        self.molecule = mol

    def test_bond_length(self):
        bond = BondLength((0, 1), self.molecule.coordinates[[0,1],])
        assert bond.value == 2.0220069632957394
        bond.set_new_coordinates(self.molecule.coordinates[[1,2],])
        assert bond.value == 3.3019199546607476

    def test_bend_angle(self):
        pass
