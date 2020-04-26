from unittest import TestCase

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from importlib_resources import path

from saddle.utils import Utils, internal_to_cartesian
from saddle.pure_internal import Bond, Angle, Dihed


class TestUtils(TestCase):

    file_list = []

    def test_load_xyz(self):
        with path("saddle.test.data", "water.xyz") as file_path:
            nums, coors, title = Utils._load_xyz(file_path)
        assert np.allclose(nums, np.array([1, 8, 1]))
        ref_coor = np.array(
            [
                [1.481237149, -0.93019116, 0.0],
                [0.0, 0.11720080, 0],
                [-1.481237149, -0.93019116, 0.0],
            ]
        )
        assert np.allclose(coors, ref_coor)
        assert title == "water"

    def test_save_xyz(self):
        with path("saddle.test.data", "water.xyz") as file_path:
            water_mol = Utils.load_file(file_path)
        with path("saddle.test.data", "") as file_path:
            new_file_name = file_path / "test_base_mole_test_file"
        Utils.save_file(new_file_name, water_mol)
        new_add_file = new_file_name.parent / (new_file_name.name + ".xyz")
        TestUtils.file_list.append(new_add_file)
        with path("saddle.test.data", "test_base_mole_test_file.xyz") as file_path:
            mol = Utils.load_file(file_path)
        ref_coor = np.array(
            [
                [1.481237149, -0.93019116, 0.0],
                [0.0, 0.11720080, 0],
                [-1.481237149, -0.93019116, 0.0],
            ]
        )
        assert np.allclose(mol.coordinates, ref_coor)
        assert np.allclose(mol.numbers, [1, 8, 1])

    def test_internal_to_cartesian_bond_angle(self):
        b1 = Bond((0, 1), 1)
        b2 = Bond((1, 2), 1)
        # test 90 degree
        a1 = Angle((0, 1, 2), np.pi / 2)
        coords = internal_to_cartesian([b1, b2, a1])
        assert_almost_equal(np.linalg.norm(coords[2] - coords[0]), np.sqrt(2))

        # test 60 degree with connected with 0
        b1 = Bond((0, 1), 1)
        b2 = Bond((0, 2), 1)
        a1 = Angle((1, 0, 2), np.pi / 3)
        coords = internal_to_cartesian([b1, b2, a1])
        assert_almost_equal(np.linalg.norm(coords[1] - coords[0]), 1)

        # test 60 degree with connected with 1
        b1 = Bond((0, 1), 1)
        b2 = Bond((1, 2), 1)
        a1 = Angle((0, 1, 2), np.pi / 3)
        coords = internal_to_cartesian([b1, b2, a1])
        assert_almost_equal(np.linalg.norm(coords[2] - coords[0]), 1)

    def test_internal_to_cartesian_all(self):
        b1 = Bond((0, 1), 1)
        b2 = Bond((1, 2), 1)
        a1 = Angle((0, 1, 2), np.pi / 2)
        b3 = Bond((2, 3), 1)
        a2 = Angle((1, 2, 3), np.pi / 2)
        d1 = Dihed((0, 1, 2, 3), np.pi * 2 / 3)
        coords = internal_to_cartesian([b1, b2, a1, b3, a2, d1])
        # test section
        r_23 = coords[3] - coords[2]
        assert_almost_equal(np.linalg.norm(r_23), 1)
        r_21 = coords[1] - coords[2]
        assert_almost_equal(np.linalg.norm(r_21), 1)
        sin_angle = r_23 @ r_21 / (np.linalg.norm(r_23) * np.linalg.norm(r_21))
        assert_almost_equal(sin_angle, 0.0)
        # cos dihed value
        cos_dihed = self._compute_dihed_helper(coords)
        assert_almost_equal(cos_dihed, np.cos(np.pi * 2 / 3))

    def test_internal_to_cartesian_random(self):
        at1 = Bond((0, 1), 1.2)
        at21 = Bond((1, 2), 1.5)
        at22 = Angle((0, 1, 2), np.pi / 2)
        at31 = Bond((2, 3), 1.2)
        at32 = Angle((1, 2, 3), np.pi / 2)
        at33 = Dihed((0, 1, 2, 3), np.pi * 2 / 3)
        at41 = Bond((3, 4), 1)
        at42 = Angle((2, 3, 4), np.pi / 2)
        at43 = Dihed((1, 2, 3, 4), -np.pi * 2 / 3)
        coords = internal_to_cartesian(
            [at1, at21, at22, at31, at32, at33, at41, at42, at43]
        )
        # test dihed1
        dihed1 = self._compute_dihed_helper(coords[[0, 1, 2, 3]])
        assert_almost_equal(dihed1, np.cos(np.pi * 2 / 3))
        # test dihed2
        dihed2 = self._compute_dihed_helper(coords[[1, 2, 3, 4]])
        assert_almost_equal(dihed2, np.cos(-np.pi * 2 / 3))

    def test_internal_to_cartesian_diff_direction(self):
        at1 = Bond((0, 1), 1.2)
        at21 = Bond((1, 2), 1.5)
        at22 = Angle((0, 1, 2), np.pi / 2)
        at31 = Bond((2, 3), 1.2)
        at32 = Angle((1, 2, 3), np.pi / 2)
        at33 = Dihed((0, 1, 2, 3), np.pi * 2 / 3)
        at41 = Bond((2, 4), 1)
        at42 = Angle((1, 2, 4), np.pi / 2)
        at43 = Dihed((0, 1, 2, 4), -np.pi * 2 / 3)
        coords = internal_to_cartesian(
            [at1, at21, at22, at31, at32, at33, at41, at42, at43]
        )
        # test dihed1
        dihed1 = self._compute_dihed_helper(coords[[0, 1, 2, 3]])
        assert_almost_equal(dihed1, np.cos(np.pi * 2 / 3))
        # test dihed2
        dihed2 = self._compute_dihed_helper(coords[[0, 1, 2, 4]])
        assert_almost_equal(dihed2, np.cos(-np.pi * 2 / 3))
        # test dihed1 and dihed2 cos value same, but different value
        assert_almost_equal(dihed1, dihed2)
        assert not np.allclose(coords[3], coords[4])


    def _compute_dihed_helper(self, coords):
        """Compute dihedral between plane 0,1,2 and plane 1,2,3."""
        r_10 = (coords[0] - coords[1]) / np.linalg.norm(coords[0] - coords[1])
        r_21 = (coords[1] - coords[2]) / np.linalg.norm(coords[1] - coords[2])
        r_32 = (coords[2] - coords[3]) / np.linalg.norm(coords[2] - coords[3])
        norm1 = np.cross(r_21, r_10)
        norm2 = np.cross(r_32, r_21)
        return norm1 @ norm2

    @classmethod
    def tearDownClass(cls):
        for i in cls.file_list:
            i.unlink()
