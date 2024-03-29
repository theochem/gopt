import unittest
from copy import deepcopy

import numpy as np
from importlib_resources import path
from saddle.cartesian import Cartesian
from saddle.periodic.periodic import angstrom
from saddle.utils import Utils


class TestCartesian(unittest.TestCase):
    def setUp(self):
        with path("saddle.test.data", "water.xyz") as mol_path:
            mol = Utils.load_file(mol_path)
        self.cartesian = Cartesian(mol.coordinates, mol.numbers, 0, 1)

    def test_from_file(self):
        with path("saddle.test.data", "water.xyz") as mol_path:
            mol = Cartesian.from_file(mol_path)
        ref_coordinates = np.array(
            [
                [0.783837, -0.492236, -0.000000],
                [-0.000000, 0.062020, -0.000000],
                [-0.783837, -0.492236, -0.000000],
            ]
        )
        assert np.allclose(mol.coordinates / angstrom, ref_coordinates)
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
        assert np.allclose(self.cartesian.coordinates / angstrom, ref_coordinates)

    def test_numbers(self):
        ref_numbers = np.array([1, 8, 1])
        assert np.allclose(self.cartesian.numbers, ref_numbers)

    def test_charge_and_multi(self):
        ref_multi = 1
        ref_charge = 0
        assert self.cartesian.multi == ref_multi
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
        assert np.allclose(self.cartesian.angle_cos(0, 1, 2), ref_angle_cos)
        assert np.allclose(self.cartesian.angle(0, 1, 2), np.arccos(ref_angle_cos))

    def test_get_energy_from_fchk(self):
        with path("saddle.test.data", "water_1.fchk") as fchk_path:
            mole = deepcopy(self.cartesian)
            mole.energy_from_fchk(fchk_path)
        assert np.allclose(mole.energy, -7.599264122862e1)
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
        assert np.allclose(mole.energy_gradient, ref_gradient)
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
        assert np.allclose(mole.coordinates, ref_coor)
