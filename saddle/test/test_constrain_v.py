from copy import deepcopy
from unittest import TestCase

import numpy as np
from importlib_resources import path
from saddle.constrain_v import NewVspace

class TestConstrainVspace(TestCase):
    def setUp(self):
        with path('saddle.test.data', 'h2o2.xyz') as mol_file:
            self.mol = NewVspace.from_file(mol_file, charge=0, multi=1)
            self.mol.auto_select_ic()

    def test_select_ic(self):
        # (0, 1), (0, 2), (1, 3), (1, 0, 2), (0, 1, 3), (2, 0, 1, 3)
        assert len(self.mol.ic) == 6

        self.mol.select_freeze_ic(1)
        # (0, 2), (0, 1), (1, 3), (1, 0, 2), (0, 1, 3), (2, 0, 1, 3)
        assert self.mol.ic[1].atoms == (0, 1)

        self.mol.select_freeze_ic(2, 4)
        # (1, 3), (0, 1, 3), (0, 2), (1, 0, 2), (0, 1), (2, 0, 1, 3)
        assert self.mol.ic[0].atoms == (1, 3)
        assert self.mol.ic[1].atoms == (0, 1, 3)

        self.mol.select_key_ic(3)
        # (1, 3), (0, 1, 3), (1, 0, 2), (0, 2), (0, 1), (2, 0, 1, 3)
        assert self.mol.ic[2].atoms == (1, 0, 2)

        self.mol.select_freeze_ic(2)
        # (1, 0, 2), (0, 1, 3), (1, 3), (0, 2), (0, 1), (2, 0, 1, 3)
        assert self.mol.ic[0].atoms == (1, 0, 2)
        assert self.mol.n_freeze == 1
        assert self.mol.n_key == 0

        self.mol.select_key_ic(4, 5)
        # (1, 0, 2), (0, 1), (2, 0, 1, 3), (0, 2), (0, 1, 3), (1, 3)
        assert self.mol.ic[2].atoms == (2, 0, 1, 3)
        assert self.mol.ic[1].atoms == (0, 1)
        assert self.mol.n_freeze == 1
        assert self.mol.n_key == 2

        self.mol.select_freeze_ic(0, 3)
        # (1, 0, 2), (0, 2), (2, 0, 1, 3), (0, 1), (0, 1, 3), (1, 3)
        assert self.mol.ic[2].atoms == (2, 0, 1, 3)
        assert self.mol.ic[1].atoms == (0, 2)
        assert self.mol.n_key == 0
        assert self.mol.n_freeze == 2
        self.mol.select_key_ic(2)
        # (1, 0, 2), (0, 2), (2, 0, 1, 3), (0, 1), (0, 1, 3), (1, 3)
        assert self.mol.ic[2].atoms == (2, 0, 1, 3)
        assert self.mol.n_key == 1
        self.mol.select_key_ic(3, 5)
        # (1, 0, 2), (0, 2), (0, 1), (1, 3), (0, 1, 3), (2, 0, 1, 3)
        assert self.mol.ic[3].atoms == (1, 3)
        assert self.mol.ic[2].atoms == (0, 1)
        assert self.mol.n_key == 2

    def test_select_ic_errors(self):
        # (0, 1), (0, 2), (1, 3), (1, 0, 2), (0, 1, 3), (2, 0, 1, 3)
        self.mol.select_key_ic(1, 2)
        # (0, 2), (1, 3), (0, 1), (1, 0, 2), (0, 1, 3), (2, 0, 1, 3)
        self.mol.select_freeze_ic(1)
        # (1, 3), (0, 2), (0, 1), (1, 0, 2), (0, 1, 3), (2, 0, 1, 3)
        with self.assertRaises(IndexError):
            self.mol.select_freeze_ic(0, -1)
        with self.assertRaises(IndexError):
            self.mol.select_freeze_ic(2, 1, -3)
        with self.assertRaises(IndexError):
            self.mol.select_freeze_ic(3, 6)

        self.mol.select_freeze_ic(0, 1, 2)
        with self.assertRaises(IndexError):
            self.mol.select_key_ic(1)
        with self.assertRaises(IndexError):
            self.mol.select_key_ic(2)
        with self.assertRaises(IndexError):
            self.mol.select_key_ic(-1, -2)
        with self.assertRaises(IndexError):
            self.mol.select_key_ic(6)

    def test_unit_vector(self):
        uni_v = self.mol._reduced_unit_vectors(3)
        ref = np.zeros((6, 3))
        ref[:3, :3] = np.eye(3)
        assert np.allclose(uni_v, ref)

        uni_v_2 = self.mol._reduced_unit_vectors(1, 4)
        ref_2 = np.zeros((6, 3))
        ref_2[1:4, :] = np.eye(3)
        assert np.allclose(uni_v_2, ref_2)

        uni_v_3 = self.mol._reduced_unit_vectors(2, 3)
        ref_3 = np.array([0, 0, 1, 0, 0, 0]).reshape(6, -1)
        assert np.allclose(uni_v_3, ref_3)

        with self.assertRaises(ValueError):
            self.mol._reduced_perturbation(0)

        with self.assertRaises(ValueError):
            self.mol._reduced_perturbation(-2)

    def test_reduced_perturbation(self):
        red_pert = self.mol._reduced_perturbation(2)
        project = np.dot(self.mol.b_matrix, np.linalg.pinv(self.mol.b_matrix))
        vec_a = np.array([1, 0, 0, 0, 0, 0])
        vec_b = np.array([0, 1, 0, 0, 0, 0])
        init_v = np.vstack((vec_a, vec_b)).T
        ref_pert = np.dot(project, np.dot(project, init_v))
        assert np.allclose(red_pert, ref_pert)

        red_pert2 = self.mol._reduced_perturbation(1, 3)
        project = np.dot(self.mol.b_matrix, np.linalg.pinv(self.mol.b_matrix))
        vec_a2 = np.array([0, 1, 0, 0, 0, 0])
        vec_b2 = np.array([0, 0, 1, 0, 0, 0])
        init_v2 = np.vstack((vec_a2, vec_b2)).T
        ref_pert2 = np.dot(project, np.dot(project, init_v2))
        assert np.allclose(red_pert2, ref_pert2)

        red_pert3 = self.mol._reduced_perturbation(2, 3)
        project = np.dot(self.mol.b_matrix, np.linalg.pinv(self.mol.b_matrix))
        init_v3 = np.array([0, 0, 1, 0, 0, 0]).reshape(-1, 1)
        ref_pert3 = np.dot(project, np.dot(project, init_v3))
        assert np.allclose(red_pert3, ref_pert3)

        with self.assertRaises(ValueError):
            self.mol._reduced_perturbation(0)
        with self.assertRaises(ValueError):
            self.mol._reduced_perturbation(-1, -3)
