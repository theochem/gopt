from copy import deepcopy
from unittest import TestCase

import numpy as np
from importlib_resources import path
from saddle.constrain_v import NewVspace
from copy import deepcopy


class TestConstrainVspace(TestCase):
    def setUp(self):
        with path('saddle.test.data', 'h2o2.xyz') as mol_file:
            self.mol = NewVspace.from_file(mol_file, charge=0, multi=1)
            self.mol.auto_select_ic()
        with path('saddle.test.data', 'water.xyz') as mol_file:
            self.mol2 = NewVspace.from_file(mol_file, charge=0, multi=1)
            self.mol2.auto_select_ic()
            self.mol2.add_bond(0, 2)

    def test_select_ic(self):
        # (0, 1), (0, 2), (1, 3), (1, 0, 2), (0, 1, 3), (2, 0, 1, 3)
        assert len(self.mol.ic) == 6

        self.mol.select_freeze_ic(1)
        # (0, 2), (0, 1), (1, 3), (1, 0, 2), (0, 1, 3), (2, 0, 1, 3)
        assert self.mol.n_nonkey == 5
        assert self.mol.ic[1].atoms == (0, 1)

        self.mol.select_freeze_ic(2, 4)
        # (1, 3), (0, 1, 3), (0, 2), (1, 0, 2), (0, 1), (2, 0, 1, 3)
        assert self.mol.n_nonkey == 4
        assert self.mol.ic[0].atoms == (1, 3)
        assert self.mol.ic[1].atoms == (0, 1, 3)

        self.mol.select_key_ic(3)
        # (1, 3), (0, 1, 3), (1, 0, 2), (0, 2), (0, 1), (2, 0, 1, 3)
        assert self.mol.n_nonkey == 3
        assert self.mol.ic[2].atoms == (1, 0, 2)

        self.mol.select_freeze_ic(2)
        # (1, 0, 2), (0, 1, 3), (1, 3), (0, 2), (0, 1), (2, 0, 1, 3)
        assert self.mol.n_nonkey == 5
        assert self.mol.ic[0].atoms == (1, 0, 2)
        assert self.mol.n_freeze == 1
        assert self.mol.n_key == 0

        self.mol.select_key_ic(4, 5)
        # (1, 0, 2), (0, 1), (2, 0, 1, 3), (0, 2), (0, 1, 3), (1, 3)
        assert self.mol.n_nonkey == 3
        assert self.mol.ic[2].atoms == (2, 0, 1, 3)
        assert self.mol.ic[1].atoms == (0, 1)
        assert self.mol.n_freeze == 1
        assert self.mol.n_key == 2

        self.mol.select_freeze_ic(0, 3)
        # (1, 0, 2), (0, 2), (2, 0, 1, 3), (0, 1), (0, 1, 3), (1, 3)
        assert self.mol.n_nonkey == 4
        assert self.mol.ic[2].atoms == (2, 0, 1, 3)
        assert self.mol.ic[1].atoms == (0, 2)
        assert self.mol.n_key == 0
        assert self.mol.n_freeze == 2
        self.mol.select_key_ic(2)
        # (1, 0, 2), (0, 2), (2, 0, 1, 3), (0, 1), (0, 1, 3), (1, 3)
        assert self.mol.n_nonkey == 3
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
            self.mol._reduced_perturbation(-1, -3)

    def test_generate_freeze_space(self):
        self.mol2.select_freeze_ic(0, 3)
        self.mol2._generate_freeze_space()
        vec_a2 = np.array([1, 0, 0, 0])
        vec_b2 = np.array([0, 1, 0, 0])
        init_v2 = np.vstack((vec_a2, vec_b2)).T
        proj = np.dot(self.mol2.b_matrix, np.linalg.pinv(self.mol2.b_matrix))
        mtx = np.dot(proj, init_v2)
        # print(self.mol2._freeze_space)
        assert mtx.shape == (4, 2)
        _, vectors = np.linalg.eigh(np.dot(mtx.T, mtx))
        bset = np.dot(mtx, vectors)
        bset[:, 0] /= np.linalg.norm(bset[:, 0])
        bset[:, 1] /= np.linalg.norm(bset[:, 1])
        assert np.allclose(self.mol2._freeze_space**2, bset**2)

    def test_generate_key_space(self):
        ref_mol = deepcopy(self.mol2)
        self.mol2.select_freeze_ic(0, 3)
        self.mol2.select_key_ic(3)
        self.mol2._generate_freeze_space()
        self.mol2._generate_key_space()
        f_space = self.mol2._freeze_space

        ref_mol.select_freeze_ic(0, 3)
        ref_mol.select_freeze_ic(0, 1, 3)
        ref_mol._generate_freeze_space()
        rf_space = ref_mol._freeze_space
        left_over_v = rf_space - np.dot(np.dot(f_space, f_space.T), rf_space)
        values, vectors = np.linalg.eigh(np.dot(left_over_v, left_over_v.T))
        ref_kspace = vectors[:, np.abs(values) > 1e-6]
        assert np.allclose(self.mol2._key_space[:, 0]**2, ref_kspace[:, 0]**2)

    def test_generate_non_space(self):
        ref_mol = deepcopy(self.mol2)
        self.mol2.select_freeze_ic(0, 3)
        self.mol2.select_key_ic(3)
        self.mol2._generate_freeze_space()
        self.mol2._generate_key_space()
        self.mol2._generate_non_space()
        f_space = self.mol2._freeze_space
        k_space = self.mol2._key_space

        ref_mol.select_freeze_ic(0, 3)
        ref_mol.select_freeze_ic(0, 1, 2, 3)
        ref_mol._generate_freeze_space()
        rf_space = ref_mol._freeze_space
        left_ov_1 = rf_space - np.dot(np.dot(f_space, f_space.T), rf_space)
        left_ov_2 = left_ov_1 - np.dot(np.dot(k_space, k_space.T), left_ov_1)
        values, vectors = np.linalg.eigh(np.dot(left_ov_2, left_ov_2.T))
        ref_n_space = vectors[:, np.abs(values) > 1e-6]
        assert np.allclose(self.mol2._non_space**2, ref_n_space**2)

    def test_vspace(self):
        self.mol2.select_freeze_ic(0, 2)
        self.mol2.select_key_ic(2)
        self.mol2.vspace
        ref_v = np.hstack((self.mol2._freeze_space, self.mol2._key_space,
                           self.mol2._non_space))
        assert np.allclose(self.mol2.vspace, ref_v)

        self.mol2.select_freeze_ic(0)
        assert not np.allclose(self.mol2.vspace, ref_v)
        vspace = self.mol2.vspace
        self.mol2.select_key_ic(1, 3)
        assert not np.allclose(self.mol2.vspace, vspace)

    def test_vspace_forms(self):
        ref_mol = deepcopy(self.mol2)
        # calculate two different vspace
        self.mol2.select_freeze_ic(0, 2)
        vs1 = self.mol2.vspace

        ref_mol.select_key_ic(0, 2)
        vs2 = ref_mol.vspace
        assert np.allclose(vs1**2, vs2**2)

    def test_set_vspace(self):
        ref_mol = deepcopy(self.mol2)
        self.mol2.select_freeze_ic(0, 2)
        self.mol2.select_key_ic(2)
        vspace = self.mol2.vspace

        ref_mol.set_vspace(vspace)
        assert np.allclose(ref_mol.vspace, self.mol2.vspace)
        assert np.allclose(ref_mol.vspace, ref_mol._non_space)

        # random vspace, set freeze, key ic values
        ref_mol._n_freeze_ic = 2
        ref_mol._n_key_ic = 1
        ref_vs = np.random.rand(4, 3)
        ref_mol.set_vspace(ref_vs)
        assert np.allclose(ref_mol._freeze_space, ref_vs[:, :2])
        assert np.allclose(ref_mol._key_space, ref_vs[:, 2:])

    def test_vs_align_matrix(self):
        ref_v = deepcopy(self.mol2.vspace)
        self.mol2.select_freeze_ic(0)
        assert self.mol2.vspace.shape == (4, 3)
        assert self.mol2.b_matrix.shape == (4, 9)
        self.mol2.select_key_ic(2)
        assert not np.allclose(self.mol2.vspace, ref_v)
        # construct ref vs
        self.mol2.align_vspace_matrix(ref_v)
        assert np.allclose(self.mol2.vspace, ref_v)
        assert np.allclose(self.mol2._freeze_space, ref_v[:, 0].reshape(4, -1))
        assert np.allclose(self.mol2._key_space, ref_v[:, 1].reshape(4, -1))
        assert np.allclose(self.mol2._non_space, ref_v[:, 2:])

        # change key_ic
        self.mol2.select_freeze_ic(0, 1)
        self.mol2.select_key_ic(2)
        assert not np.allclose(self.mol2.vspace, ref_v)
        self.mol2.align_vspace_matrix(ref_v)
        assert np.allclose(self.mol2.vspace, ref_v)
        assert np.allclose(self.mol2._freeze_space, ref_v[:, :2])
        assert np.allclose(self.mol2._key_space, ref_v[:, 2:].reshape(4, -1))
        assert np.allclose(self.mol2._non_space, ref_v[:, 3:])

        # use separate align
        self.mol2.select_freeze_ic(0, 1)
        self.mol2.select_key_ic(2)
        assert not np.allclose(self.mol2.vspace, ref_v)
        self.mol2.align_vspace_matrix(ref_v, special=True)
        assert np.allclose(self.mol2.vspace, ref_v)
        assert np.allclose(self.mol2._freeze_space, ref_v[:, :2])
        assert np.allclose(self.mol2._key_space, ref_v[:, 2:].reshape(4, -1))
        assert np.allclose(self.mol2._non_space, ref_v[:, 3:])


    def test_vs_align_object(self):
        ref_mol = deepcopy(self.mol2)
        self.mol2.select_freeze_ic(0)
        self.mol2.select_key_ic(2)
        assert not np.allclose(self.mol2.vspace, ref_mol.vspace)
        self.mol2.align_vspace(ref_mol)
        assert np.allclose(self.mol2.vspace, ref_mol.vspace)
        with self.assertRaises(ValueError):
            self.mol2.align_vspace(ref_mol, ic_check=True)
        with self.assertRaises(TypeError):
            self.mol2.align_vspace(ref_mol.vspace)
