from unittest import TestCase

import numpy as np
from importlib_resources import path
from saddle.utils import Utils


class TestBaseMole(TestCase):

    file_list = []

    def test_load_xyz(self):
        with path('saddle.test.data', 'water.xyz') as file_path:
            nums, coors, title = Utils._load_xyz(file_path)
        assert np.allclose(nums, np.array([1, 8, 1]))
        ref_coor = np.array([[1.481237149, -0.93019116,
                              0.], [0., 0.11720080, 0],
                             [-1.481237149, -0.93019116, 0.]])
        assert np.allclose(coors, ref_coor)
        assert title == 'water'

    def test_save_xyz(self):
        with path('saddle.test.data', 'water.xyz') as file_path:
            water_mol = Utils.load_file(file_path)
        with path('saddle.test.data', '') as file_path:
            new_file_name = file_path / 'test_base_mole_test_file'
        water_mol.save_file(new_file_name)
        new_add_file = new_file_name.parent / (new_file_name.name + '.xyz')
        TestBaseMole.file_list.append(new_add_file)
        with path('saddle.test.data',
                  'test_base_mole_test_file.xyz') as file_path:
            mol = Utils.load_file(file_path)
        ref_coor = np.array([[1.481237149, -0.93019116,
                              0.], [0., 0.11720080, 0],
                             [-1.481237149, -0.93019116, 0.]])
        assert np.allclose(mol.coordinates, ref_coor)
        assert np.allclose(mol.numbers, [1, 8, 1])

    @classmethod
    def tearDownClass(cls):
        for i in cls.file_list:
            i.unlink()
