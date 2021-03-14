from unittest import TestCase
from copy import deepcopy
from importlib.resources import path

import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_almost_equal, assert_allclose
from gopt.utils import Utils, Z2C


class TestUtils(TestCase):

    file_list = []

    def test_load_xyz(self):
        with path("gopt.test.data", "water.xyz") as file_path:
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
        with path("gopt.test.data", "water.xyz") as file_path:
            water_mol = Utils.load_file(file_path)
        with path("gopt.test.data", "") as file_path:
            new_file_name = file_path / "test_base_mole_test_file"
        Utils.save_file(new_file_name, water_mol)
        new_add_file = new_file_name.parent / (new_file_name.name + ".xyz")
        TestUtils.file_list.append(new_add_file)
        with path("gopt.test.data", "test_base_mole_test_file.xyz") as file_path:
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

    def test_internal_to_cartesian(self):
        """Test internal coordinates (Zmt style) to xyz."""
        z = Z2C()
        assert z.natom == 0
        b1 = 0.8
        b2 = 1.3
        b3 = 1.5
        ang1 = np.pi / 3
        ang2 = np.pi / 4
        dihed1 = np.pi / 5
        # add first bond
        z.add_z_entry([0, 1], [b1])
        assert_allclose(z.coords[0], [0.0, 0.0, 0.0])
        assert_allclose(z.coords[1], [0.0, 0.0, b1])
        assert z.natom == 2

        # add third bond
        z.add_z_entry([0, 1, 2], [b2, ang1])
        b_10 = z.coords[0] - z.coords[1]
        b_12 = z.coords[2] - z.coords[1]
        assert_allclose(norm(b_10), b1)
        assert_allclose(norm(b_12), b2)
        assert_almost_equal(b_10 @ b_12 / norm(b_12) / norm(b_10), np.cos(ang1))
        assert z.natom == 3

        # add the fourth atom
        z.add_z_entry([0, 1, 2, 3], [b3, ang2, dihed1])
        # print(z.coords)
        b_10 = z.coords[0] - z.coords[1]
        b_12 = z.coords[2] - z.coords[1]
        b_23 = z.coords[3] - z.coords[2]
        assert_allclose(norm(b_10), b1)
        assert_allclose(norm(b_12), b2)
        assert_allclose(norm(b_23), b3)
        assert_almost_equal(b_10 @ b_12 / norm(b_12) / norm(b_10), np.cos(ang1))
        assert_almost_equal(
            -b_12 @ b_23 / norm(b_12) / norm(b_23), np.cos(ang2),
        )
        test_dihed = self._compute_dihed_helper(z.coords)
        assert_almost_equal(test_dihed, np.cos(dihed1))
        assert z.natom == 4

        # add the fifth atom, check opposite direction
        z.add_z_entry([0, 1, 2, 4], [b3, ang2, -dihed1])
        b_10 = z.coords[0] - z.coords[1]
        b_12 = z.coords[2] - z.coords[1]
        b_24 = z.coords[4] - z.coords[2]
        assert_allclose(norm(b_10), b1)
        assert_allclose(norm(b_12), b2)
        assert_allclose(norm(b_24), b3)
        assert_almost_equal(b_10 @ b_12 / norm(b_12) / norm(b_10), np.cos(ang1))
        assert_almost_equal(
            -b_12 @ b_24 / norm(b_12) / norm(b_24), np.cos(ang2),
        )
        test_dihed = self._compute_dihed_helper(z.coords)
        assert_almost_equal(test_dihed, np.cos(-dihed1))
        assert z.natom == 5
        assert not np.allclose(z.coords[3], z.coords[4])

    def test_random_internal(self):
        """Generate random bond, angle, dihed."""
        for i in range(10):
            z = Z2C()
            assert z.natom == 0
            b1, b2, b3 = np.random.rand(3) * 2
            ang1, ang2 = np.random.rand(2) * np.pi
            dihed1 = (np.random.rand(1)[0] - 0.5) * 4 * np.pi
            # add first bond
            z.add_z_entry([0, 1], [b1])
            assert_allclose(z.coords[0], [0.0, 0.0, 0.0])
            assert_allclose(z.coords[1], [0.0, 0.0, b1])
            assert z.natom == 2

            # add third bond
            z.add_z_entry([0, 1, 2], [b2, ang1])
            b_10 = z.coords[0] - z.coords[1]
            b_12 = z.coords[2] - z.coords[1]
            assert_allclose(norm(b_10), b1)
            assert_allclose(norm(b_12), b2)
            assert_almost_equal(b_10 @ b_12 / norm(b_12) / norm(b_10), np.cos(ang1))
            assert z.natom == 3

            # add the fourth atom
            z.add_z_entry([0, 1, 2, 3], [b3, ang2, dihed1])
            # print(z.coords)
            b_10 = z.coords[0] - z.coords[1]
            b_12 = z.coords[2] - z.coords[1]
            b_23 = z.coords[3] - z.coords[2]
            assert_allclose(norm(b_10), b1)
            assert_allclose(norm(b_12), b2)
            assert_allclose(norm(b_23), b3)
            assert_almost_equal(b_10 @ b_12 / norm(b_12) / norm(b_10), np.cos(ang1))
            assert_almost_equal(
                -b_12 @ b_23 / norm(b_12) / norm(b_23), np.cos(ang2),
            )
            # print(z.coords)
            test_dihed = self._compute_dihed_helper(z.coords)
            assert_almost_equal(test_dihed, np.cos(dihed1))
            assert z.natom == 4

    def test_z2c_error(self):
        """Test error situation."""
        z = Z2C()
        with self.assertRaises(ValueError):
            z.add_z_entry([0, 1], [1.1, 1.2])
        with self.assertRaises(ValueError):
            z.add_z_entry([0, 1], [-0.5])
        with self.assertRaises(ValueError):
            z.add_z_entry([0, 1, 2], [0.5, 1])
        z.add_z_entry([0, 1], [1.0])
        with self.assertRaises(ValueError):
            z.add_z_entry([1, 2, 3], [0.5, 1.5 * np.pi])
        with self.assertRaises(ValueError):
            z.add_z_entry([1, 2, 3, 4], [0.5, 0.5 * np.pi, 1.0])
        z.add_z_entry([0, 1, 2], [1, np.pi / 2])
        with self.assertRaises(ValueError):
            z.add_z_entry([1, 2, 3], [0.5, 0.5 * np.pi])

    def _compute_dihed_helper(self, coords):
        """Compute dihedral between plane 0,1,2 and plane 1,2,3."""
        r_10 = (coords[0] - coords[1]) / norm(coords[0] - coords[1])
        r_21 = (coords[1] - coords[2]) / norm(coords[1] - coords[2])
        r_32 = (coords[2] - coords[3]) / norm(coords[2] - coords[3])
        norm_vec1 = np.cross(r_21, r_10)
        norm_vec2 = np.cross(r_32, r_21)
        norm_vec1 /= norm(norm_vec1)
        norm_vec2 /= norm(norm_vec2)
        return norm_vec1 @ norm_vec2

    @classmethod
    def tearDownClass(cls):
        for i in cls.file_list:
            i.unlink()
