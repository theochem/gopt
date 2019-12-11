import unittest

import numpy as np
from importlib_resources import path
from numpy.testing import assert_allclose
from saddle.math_lib import diagonalize
from saddle.ts_construct import TSConstruct


class TestPathRI(unittest.TestCase):
    @classmethod
    def setup_class(self):
        with path("saddle.test.data", "rct.xyz") as rct_path:
            with path("saddle.test.data", "prd.xyz") as prd_path:
                self.ts_mol = TSConstruct.from_file(rct_path, prd_path)
        self.ts_mol.auto_generate_ts(task="path")
        self.path_mol = self.ts_mol.ts

    def test_basic_property(self):
        assert self.path_mol.key_ic_number == 0
        try:
            self.path_mol.set_key_ic_number(2)
        except NotImplementedError:
            pass
        assert self.path_mol.key_ic_number == 0

        diff = self.ts_mol.prd.ic_values - self.ts_mol.rct.ic_values
        assert np.allclose(self.path_mol.path_vector, diff)

        # genreate proper real unit path vector
        real_path_v = (
            self.ts_mol.ts.b_matrix @ np.linalg.pinv(self.ts_mol.ts.b_matrix) @ diff
        )
        assert np.allclose(
            real_path_v / np.linalg.norm(real_path_v),
            self.ts_mol.ts.real_unit_path_vector,
        )

        # dim space of v is one less than degree of freedom
        assert self.ts_mol.ts.vspace.shape[1] == self.ts_mol.ts.df - 1

    def test_v_space(self):
        self.path_mol._generate_reduce_space()
        assert np.linalg.norm(self.path_mol._red_space) - 1 < 1e-8
        diff = np.dot(self.path_mol.vspace.T, self.path_mol.path_vector)
        assert np.linalg.norm(diff) < 1e-8
        self.path_mol._reset_v_space()
        assert self.path_mol._vspace is None
        assert self.path_mol.key_ic_number == 0
        diff = np.dot(self.path_mol.vspace.T, self.path_mol.path_vector)
        assert np.linalg.norm(diff) < 1e-8

    def test_v_space_change(self):
        diff = np.dot(self.path_mol.vspace.T, self.path_mol.path_vector)
        assert np.linalg.norm(diff) < 1e-8
        # construct vspace
        c = np.random.rand(9, 5)
        values, vectors = np.linalg.eigh(np.dot(c, c.T))
        basis_vects = vectors[:, np.abs(values) > 1e-5]
        # project out path_vector
        no_path_space = basis_vects - np.dot(
            np.outer(
                self.path_mol.real_unit_path_vector, self.path_mol.real_unit_path_vector
            ),
            basis_vects,
        )
        assert (
            np.linalg.norm(np.dot(no_path_space.T, self.path_mol.real_unit_path_vector))
            < 1e-8
        )
        w, v = diagonalize(no_path_space)
        new_v = v[:, np.abs(w) > 1e-5]
        assert (
            np.linalg.norm(np.dot(new_v.T, self.path_mol.real_unit_path_vector)) < 1e-8
        )
        # test all part are orthonormal
        assert_allclose(np.linalg.norm(new_v, axis=0), np.ones(c.shape[1]))

        # given a random vspace, generate a property one
        self.path_mol.set_vspace(basis_vects)
        assert_allclose(self.path_mol.vspace, new_v)
        assert_allclose(
            np.linalg.norm(self.path_mol.vspace, axis=0), np.ones(c.shape[1])
        )
