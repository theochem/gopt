import os

import numpy as np
from pkg_resources import Requirement, resource_filename

from saddle.conf import work_dir
from saddle.gaussianwrapper import GaussianWrapper
from saddle.iodata import IOData


class TestGaussWrap(object):

    path = os.path.dirname(os.path.realpath(__file__))
    file_list = []

    def setUp(self):
        mol_path = resource_filename(
            Requirement.parse('saddle'), 'data/water.xyz')
        mol = IOData.from_file(mol_path)
        self.gwob = GaussianWrapper(mol, title='water')

    def test_create_ins(self):
        assert self.gwob.title == 'water'
        assert (isinstance(self.gwob.molecule, IOData))

    def test_create_input(self):
        self.gwob.create_gauss_input(0, 1, spe_title='test_gauss')
        filepath = os.path.join(work_dir, "test_gauss.com")
        mol = IOData.from_file(filepath)
        self.file_list.append(filepath)
        assert np.allclose(self.gwob.molecule.coordinates, mol.coordinates)

    def test_create_input_gjf(self):
        path = resource_filename(Requirement.parse('saddle'), 'work')
        self.gwob.create_gauss_input(
            0, 1, spe_title='test_2nd_gauss', path=self.path, postfix='.gjf')
        filepath = os.path.join(self.path, 'test_2nd_gauss.gjf')
        self.file_list.append(filepath)
        mol = IOData.from_file(filepath)
        assert np.allclose(self.gwob.molecule.coordinates, mol.coordinates)

    def test_create_input_file(self):
        self.gwob.title = 'test_untitled'
        input_file = self.gwob._create_input_file(0, 1)
        filepath = os.path.join(work_dir, input_file + ".com")
        mol = IOData.from_file(filepath)
        self.file_list.append(filepath)
        assert np.allclose(self.gwob.molecule.coordinates, mol.coordinates)

    @classmethod
    def tearDownClass(cls):
        for i in cls.file_list:
            os.remove(i)
