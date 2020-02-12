"""Procrustes test files."""
import unittest
from collections import Iterable
from copy import deepcopy

from importlib_resources import path

import numpy as np

from saddle.periodic.periodic import periodic
from saddle.periodic.units import amu
from saddle.procrustes.procrustes import Procrustes
from saddle.utils import Utils


class test_procrustes(unittest.TestCase):
    """Procrustes test class."""

    def test_barycenter(self):
        """Test barycenter of molecule."""
        ori_numbers = np.array([1, 2])
        m1 = periodic[1].mass / amu
        m2 = periodic[2].mass / amu
        mass_numbers = np.array([m1, m2])
        assert np.allclose(mass_numbers, np.array([1, 4]), atol=1.0e-2)
        coordinates = np.array([[0, 1, 0], [1, 0, 0]])
        result = np.einsum("i, ij -> j", mass_numbers, coordinates)
        assert np.allclose(
            Procrustes._barycenter(coordinates, ori_numbers),
            result / np.sum(mass_numbers),
        )

    def test_fetch_atomic_amass(self):
        """Test get atomic mass."""
        assert Procrustes._fetch_atomic_mass(1) - 1.007975 < 1e-5
        assert Procrustes._fetch_atomic_mass(6) - 12.0106 < 1e-5

    def test_move_center(self):
        """Test move the center of molecule to another place."""
        numbers = np.array([1, 2])
        coordinates = np.array([[0, 1, 0], [1, 0, 0]])
        coor_1st = Procrustes._move_to_center(coordinates, numbers)
        coordinates_2 = np.array([[3, 1, 0], [4, 0, 0]])
        coor_2nd = Procrustes._move_to_center(coordinates_2, numbers)
        assert np.allclose(coor_1st, coor_2nd)

        np.random.seed(10)  # set random number seed
        extra = np.random.rand(3)
        coordinates_3 = coordinates + extra
        coor_3rd = Procrustes._move_to_center(coordinates_3, numbers)
        assert np.allclose(coor_1st, coor_3rd)

        extra_2 = np.random.rand(3)
        coordinates_4 = coordinates_3 + extra_2
        coor_4th = Procrustes._move_to_center(coordinates_4, numbers)
        assert np.allclose(coor_1st, coor_4th)

    def test_rotate_coordiantes(self):
        """Test rotate two coordinates to align."""
        coordinates = np.array([0, 1, 0])
        coordinates_2 = np.array([-1, 0, 0])
        coor_2 = Procrustes._rotate_coordinates(coordinates, coordinates_2)
        assert np.allclose(coordinates, coor_2)

        coordinates_2d = np.array([[1, 1, 0], [-1, 1, 0]])
        coordinates_2d_2 = np.array([[-1, 1, 0], [-1, -1, 0]])
        coor_2d_2 = Procrustes._rotate_coordinates(coordinates_2d, coordinates_2d_2)
        assert np.allclose(coordinates_2d, coor_2d_2)

        coordinates_2d_3 = np.array([[-1, 0, 1], [-1, 0, -1]])
        coor_2d_3 = Procrustes._rotate_coordinates(coordinates_2d, coordinates_2d_3)
        assert np.allclose(coordinates_2d, coor_2d_3)

    def test_move_and_rotate(self):
        """Test first move the molecule and then rotate to align."""
        numbers = np.array([1, 2])
        coordinates = np.array([[0, 1, 0], [1, 0, 0]])
        coordinates_2 = np.array([[0, 1, 1], [1, 2, 1]])
        center_1 = Procrustes._move_to_center(coordinates, numbers)
        center_2_tmp = Procrustes._move_to_center(coordinates_2, numbers)
        center_2 = Procrustes._rotate_coordinates(center_1, center_2_tmp)
        assert np.allclose(center_1, center_2)

    def test_main_function(self):
        """Test the main function and use case for procrustes."""
        # file_path = resource_filename(
        #     Requirement.parse('saddle'), 'data/water.xyz')
        with path("saddle.procrustes.test.data", "water.xyz") as file_path:
            water = Utils.load_file(file_path)
        water.coordinates = np.array([[0, 1, 0], [1, 0, 0], [-1, -1, 1]])
        water_2 = deepcopy(water)
        water_2.coordinates = np.array([[-1, 0, 0], [0, 1, 0], [1, -1, 0]])
        water_2.coordinates += 1
        water_3 = deepcopy(water)
        water_3.coordinates = np.array([[1, 0, 0], [0, -1, 0], [-1, 1, 0]])
        water_3.coordinates -= 1

        pcs = Procrustes(water, water_2, water_3)
        final_xyz = pcs.rotate_mols()
        assert isinstance(final_xyz, Iterable)
        assert len(list(final_xyz)) == 2
        for i in final_xyz:
            assert np.allclose(i, water.coordinates)
