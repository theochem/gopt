import unittest
from copy import deepcopy
from importlib.resources import path

import numpy as np
from numpy.testing import assert_allclose
from gopt.cartesian import Cartesian
from gopt.periodic.periodic import angstrom
from gopt.utils import Utils


class TestCartesian(unittest.TestCase):
    def setUp(self):
        with path("gopt.test.data", "water.xyz") as mol_path:
            mol = Utils.load_file(mol_path)
        self.cartesian = Cartesian(mol.atcoords, mol.atnums, 0, 1)

    def test_from_file(self):
        with path("gopt.test.data", "water.xyz") as mol_path:
            mol = Cartesian.from_file(mol_path)
        ref_coordinates = np.array(
            [
                [0.783837, -0.492236, -0.000000],
                [-0.000000, 0.062020, -0.000000],
                [-0.783837, -0.492236, -0.000000],
            ]
        )
        assert_allclose(mol.atcoords / angstrom, ref_coordinates)
        assert mol.natom == 3
        assert isinstance(mol, Cartesian)

    def test_coordinates(self):
        ref_coordinates = np.array(
            [
                [0.783837, -0.492236, -0.000000],
                [-0.000000, 0.062020, -0.000000],
                [-0.783837, -0.492236, -0.000000],
            ]
        )
        assert_allclose(self.cartesian.atcoords / angstrom, ref_coordinates)

    def test_numbers(self):
        ref_numbers = np.array([1, 8, 1])
        assert_allclose(self.cartesian.atnums, ref_numbers)

    def test_charge_and_multi(self):
        ref_multi = 1
        ref_charge = 0
        assert self.cartesian.spinmult == ref_multi
        assert self.cartesian.charge == ref_charge

    def test_distance(self):
        ref_distance = np.linalg.norm(
            np.array([0.783837, -0.492236, -0.000000])
            - np.array([-0.000000, 0.062020, -0.000000])
        )
        assert self.cartesian.distance(0, 1) / angstrom == ref_distance

    def test_angle(self):
        vector1 = np.array([-0.000000, 0.062020, -0.000000]) - np.array(
            [0.783837, -0.492236, -0.000000]
        )
        vector2 = np.array([-0.000000, 0.062020, -0.000000]) - np.array(
            [-0.783837, -0.492236, -0.000000]
        )
        ref_angle_cos = (
            np.dot(vector1, vector2) / np.linalg.norm(vector1) / np.linalg.norm(vector2)
        )
        assert_allclose(self.cartesian.angle_cos(0, 1, 2), ref_angle_cos)
        assert_allclose(self.cartesian.angle(0, 1, 2), np.arccos(ref_angle_cos))

    def test_get_energy_from_fchk(self):
        with path("gopt.test.data", "water_1.fchk") as fchk_path:
            mole = deepcopy(self.cartesian)
            mole.energy_from_fchk(fchk_path)
        assert_allclose(mole.energy, -7.599264122862e1)
        ref_gradient = [
            2.44329621e-17,
            4.95449892e-03,
            -9.09914286e-03,
            7.79137241e-16,
            -3.60443012e-16,
            1.81982857e-02,
            -8.03570203e-16,
            -4.95449892e-03,
            -9.09914286e-03,
        ]
        assert_allclose(mole.energy_gradient, ref_gradient)
        ref_coor = np.array(
            [
                0.00000000e00,
                1.48124293e00,
                -8.37919685e-01,
                0.00000000e00,
                3.42113883e-49,
                2.09479921e-01,
                -1.81399942e-16,
                -1.48124293e00,
                -8.37919685e-01,
            ]
        ).reshape(-1, 3)
        assert_allclose(mole.atcoords, ref_coor)
