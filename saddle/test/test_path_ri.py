import numpy as np
from pkg_resources import Requirement, resource_filename

from saddle.ts_construct import TSConstruct

import unittest


class TestPathRI(unittest.TestCase):
    @classmethod
    def setup_class(self):
        rct_path = resource_filename(
            Requirement.parse('saddle'), 'data/rct.xyz')
        prd_path = resource_filename(
            Requirement.parse('saddle'), 'data/prd.xyz')
        self.ts_mol = TSConstruct.from_file(rct_path, prd_path)
        self.ts_mol.auto_generate_ts(task='path')
        self.path_mol = self.ts_mol.ts

    def test_basic_property(self):
        assert self.path_mol.key_ic_number == 1
        try:
            self.path_mol.set_key_ic_number(0)
        except NotImplementedError:
            pass
        assert self.path_mol.key_ic_number == 1

        diff = self.ts_mol.prd.ic_values - self.ts_mol.rct.ic_values
        assert np.allclose(self.path_mol.path_vector, diff)

    def test_v_space(self):
        self.path_mol._generate_reduce_space()
        assert np.linalg.norm(self.path_mol._red_space) - 1 < 1e-8
        diff = np.dot(self.path_mol.vspace.T, self.path_mol.path_vector)
        assert np.linalg.norm(diff) < 1e-8
        self.path_mol._reset_v_space()
        assert self.path_mol._vspace is None
        assert self.path_mol.key_ic_number == 1
        diff = np.dot(self.path_mol.vspace.T, self.path_mol.path_vector)
        assert np.linalg.norm(diff) < 1e-8
