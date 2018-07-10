import os
import unittest

import numpy as np
from importlib_resources import path
from saddle.conf import work_dir
from saddle.gaussianwrapper import GaussianWrapper
from saddle.utils import Utils


class TestGaussWrap(unittest.TestCase):

    path = os.path.dirname(os.path.realpath(__file__))
    file_list = []

    def setUp(self):
        with path('saddle.test.data', 'water.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        self.gwob = GaussianWrapper(mol, title='water')

    def test_create_ins(self):
        assert self.gwob.title == 'water'
        assert (isinstance(self.gwob.molecule, Utils))

    def test_create_input(self):
        self.gwob.create_gauss_input(0, 1, spe_title='test_gauss')
        filepath = os.path.join(work_dir, "test_gauss.com")
        mol = Utils.load_file(filepath)
        self.file_list.append(filepath)
        assert np.allclose(self.gwob.molecule.coordinates, mol.coordinates)

    def test_create_input_gjf(self):
        self.gwob.create_gauss_input(
            0, 1, spe_title='test_2nd_gauss', path=self.path, postfix='.gjf')
        filepath = os.path.join(self.path, 'test_2nd_gauss.gjf')
        self.file_list.append(filepath)
        mol = Utils.load_file(filepath)
        assert np.allclose(self.gwob.molecule.coordinates, mol.coordinates)

    def test_create_input_file(self):
        self.gwob.title = 'test_untitled'
        input_file = self.gwob._create_input_file(0, 1)
        filepath = os.path.join(work_dir, input_file + ".com")
        mol = Utils.load_file(filepath)
        self.file_list.append(filepath)
        assert np.allclose(self.gwob.molecule.coordinates, mol.coordinates)

    @classmethod
    def tearDownClass(cls):
        for i in cls.file_list:
            os.remove(i)
