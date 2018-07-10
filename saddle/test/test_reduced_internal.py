from copy import deepcopy
from unittest import TestCase

import numpy as np
from importlib_resources import path
from saddle.internal import Internal
from saddle.reduced_internal import ReducedInternal
from saddle.utils import Utils


class TestReduceInternal(TestCase):
    def setUp(self):
        with path('saddle.test.data', 'water.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        self.red_int = ReducedInternal(mol.coordinates, mol.numbers, 0, 1,
                                       'water')
        self.red_int.add_bond(1, 0)
        self.red_int.add_bond(1, 2)
        self.red_int.add_bond(0, 2)
        self.red_int.add_angle(0, 1, 2)
        self.red_int.add_angle(1, 0, 2)
        self.red_int.set_key_ic_number(2)

    def test_internal_reset_property(self):
        sample_mol = deepcopy(self.red_int)
        sample_mol.add_angle(0, 2, 1)
        assert len(sample_mol.ic) == 6
        assert self.red_int.vspace.shape == (5, 3)
        self.red_int.set_new_ics(sample_mol.ic)
        assert self.red_int.vspace.shape == (6, 3)
        self.red_int.auto_select_ic()
        assert self.red_int.vspace.shape == (3, 3)
        self.red_int.wipe_ic_info(
            I_am_sure_i_am_going_to_wipe_all_ic_info=True)
        assert self.red_int._vspace is None
        assert self.red_int._red_space is None
        assert self.red_int._non_red_space is None
        self.red_int.add_bond(0, 1)
        assert len(self.red_int.ic) == 1
        self.red_int.add_bond(2, 1)
        self.red_int.add_angle(0, 1, 2)
        assert self.red_int.vspace.shape == (3, 3)
        assert len(self.red_int.ic) == 3
        self.red_int.add_angle(0, 2, 1)
        self.red_int.add_angle(1, 0, 2)
        assert self.red_int.vspace.shape == (3, 3)
        self.red_int.add_bond(0, 2)
        self.red_int.delete_ic(0)
        assert self.red_int.vspace.shape == (3, 3)
        cache_v = self.red_int.vspace.copy()
        self.red_int.set_new_coordinates(
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        assert not np.allclose(cache_v, self.red_int.vspace)
        cache_v = self.red_int.vspace
        cache_ic = self.red_int.ic_values.copy()
        self.red_int.swap_internal_coordinates(0, 1)
        assert not np.allclose(cache_v, self.red_int.vspace)
        assert cache_ic[0] == self.red_int.ic_values[1]
        assert cache_ic[1] == self.red_int.ic_values[0]
        # test title
        assert self.red_int._title == 'water'

    def test_property(self):
        assert self.red_int.key_ic_number == 2
        assert len(self.red_int.ic) == 5
        assert isinstance(self.red_int, Internal)
        assert self.red_int.df == 3

    def test_reduce_coordinates(self):
        mole = self.red_int
        assert np.allclose(
            mole.ic_values,
            [1.81413724, 1.81413724, 2.96247453, 1.91063401, 0.61547931])
        mole.select_key_ic(0, 2)
        assert np.allclose(
            mole.ic_values,
            [1.81413724, 2.96247453, 1.81413724, 1.91063401, 0.61547931])
        svd = mole._svd_of_b_matrix()
        ref_vectors = np.array(
            [[6.96058989e-01, -3.20802187e-01, -4.62909977e-01], [
                -2.70878427e-01, -2.12077861e-01, -7.55929035e-01
            ], [-2.53716249e-01, 6.67123979e-01, -4.62909977e-01],
             [-6.03451079e-01, -4.72457758e-01, 1.97531517e-15],
             [1.16625994e-01, 4.28763572e-01, -1.77057094e-15]])
        for i in range(svd.shape[1]):
            assert (np.allclose(svd[:, i], ref_vectors[:, i])
                    or np.allclose(svd[:, i], -1 * ref_vectors[:, i]))

        ref_unit = np.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.], [0., 0.]])
        assert np.allclose(mole._reduced_unit_vectors(), ref_unit)
        ptrb = mole._reduced_perturbation()
        ref_vec = np.dot(mole.b_matrix,
                         np.dot(np.linalg.pinv(mole.b_matrix), ref_unit))
        assert np.allclose(ref_vec, ptrb)
        ref_red_space = np.array([
            [0.44093001, -0.77928078],  # reduced space
            [-0.56140826, -0.61204707],
            [-0.45792057, -0.03282547],
            [-0.52276081, 0.04872587],
            [0.08620545, 0.12111205],
        ])
        mole._generate_reduce_space()
        assert np.allclose(mole._red_space, ref_red_space)
        ref_non_red_space = np.array([
            [6.12072474e-17],  # nonreduced space
            [-3.33066907e-16],
            [-7.16200549e-01],
            [5.58315734e-01],
            [-4.18736570e-01],
        ])
        mole._generate_nonreduce_space()
        assert (np.allclose(ref_non_red_space, mole._non_red_space)
                or np.allclose(ref_non_red_space, -1 * mole._non_red_space))
        vp_ref = np.hstack((ref_red_space, ref_non_red_space))
        mole._reset_v_space()
        for i in range(vp_ref.shape[1]):
            assert (np.allclose(mole.vspace[:, i], vp_ref[:, i])
                    or np.allclose(mole.vspace[:, i], -1 * vp_ref[:, i]))

    def test_ic_ric_transform(self):
        with path('saddle.test.data', 'water.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        ri_mol = Internal(mol.coordinates, mol.numbers, 0, 1)
        ri_mol.add_bond(1, 0)
        ri_mol.add_bond(1, 2)
        ri_mol.add_bond(0, 2)
        ri_mol.add_angle(0, 1, 2)
        ri_mol.add_angle(1, 0, 2)
        vc_ref = np.array(
            [1.81413724, 1.81413724, 2.96247453, 1.91063401, 0.61547931])
        assert np.allclose(ri_mol.ic_values, vc_ref)
        ri_mol = ReducedInternal.update_to_reduced_internal(ri_mol)
        assert isinstance(ri_mol, ReducedInternal)
        ri_mol.set_key_ic_number(2)
        ri_mol.select_key_ic(0, 2)
        print(ri_mol.vspace)
        vp_ref = np.array([[4.40930006e-01, -7.79280781e-01, 6.12072474e-17], [
            -5.61408260e-01, -6.12047068e-01, -3.33066907e-16
        ], [-4.57920570e-01, -3.28254718e-02, -7.16200549e-01],
                           [-5.22760813e-01, 4.87258745e-02, 5.58315734e-01],
                           [8.62054537e-02, 1.21112047e-01, -4.18736570e-01]])
        for i in range(vp_ref.shape[1]):
            assert (np.allclose(ri_mol.vspace[:, i], vp_ref[:, i])
                    or np.allclose(ri_mol.vspace[:, i], -1 * vp_ref[:, i]))
        ri_mol.set_key_ic_number(1)
        assert ri_mol._red_space is None
        assert ri_mol._non_red_space is None
        ri_mol.set_key_ic_number(2)
        for i in range(vp_ref.shape[1]):
            assert (np.allclose(ri_mol.vspace[:, i], vp_ref[:, i])
                    or np.allclose(ri_mol.vspace[:, i], -1 * vp_ref[:, i]))
        new_coor = np.array([[1.40, -0.93019123, -0.], [-0., 0.11720081, -0.],
                             [-1.40, -0.93019123, -0.]])
        ri_mol.set_new_coordinates(new_coor)
        ref_ic = [
            1.7484364736491811, 2.8, 1.7484364736491811, 1.8569769819,
            0.6423078258
        ]
        assert np.allclose(ri_mol.ic_values, ref_ic)
        ri_mol.vspace
        assert ri_mol._red_space is not None
        assert ri_mol._non_red_space is not None
        ri_mol.add_angle(0, 2, 1)
        assert ri_mol._red_space is None
        assert ri_mol._non_red_space is None

    def test_ric_add_ic(self):
        with path('saddle.test.data', 'water.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        ri_mol = Internal(mol.coordinates, mol.numbers, 0, 1)
        ri_mol = ReducedInternal.update_to_reduced_internal(ri_mol)
        ri_mol.add_bond(1, 0)
        ri_mol.add_bond(1, 2)
        ri_mol.add_bond(0, 2)
        ri_mol.vspace
        ri_mol.add_angle(0, 1, 2)
        ri_mol.add_angle(1, 0, 2)
        ri_mol.select_key_ic(0, 2)
        vp_ref = np.array([[4.40930006e-01, -7.79280781e-01, 6.12072474e-17], [
            -5.61408260e-01, -6.12047068e-01, -3.33066907e-16
        ], [-4.57920570e-01, -3.28254718e-02, -7.16200549e-01],
                           [-5.22760813e-01, 4.87258745e-02, 5.58315734e-01],
                           [8.62054537e-02, 1.21112047e-01, -4.18736570e-01]])
        for i in range(vp_ref.shape[1]):
            assert (np.allclose(ri_mol.vspace[:, i], vp_ref[:, i])
                    or np.allclose(ri_mol.vspace[:, i], -1 * vp_ref[:, i]))
        ri_mol.set_key_ic_number(1)
        assert ri_mol._red_space is None
        assert ri_mol._non_red_space is None

    def test_get_delta_v(self):
        with path('saddle.test.data', 'water.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        ri_mol = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        ri_mol.add_bond(1, 0)
        ri_mol.add_bond(1, 2)
        ri_mol.add_angle(0, 1, 2)
        ri_mol.set_key_ic_number(1)
        vp_ref = np.array([[-1.00000000e+00, 1.52461867e-16, -2.25191203e-16],
                           [-1.57750703e-16, 3.51986473e-01, 9.36005087e-01],
                           [-1.73472348e-16, -9.36005087e-01, 3.51986473e-01]])
        ri_mol.set_vspace(vp_ref)
        print(ri_mol.vspace)
        assert np.allclose(ri_mol.vspace, vp_ref)
        v_change = np.array([-0.2, 0.07039729, 0.18720102])
        ic_change = np.dot(ri_mol.vspace, v_change)
        assert np.allclose(ic_change, np.array([0.2, 0.2, 0.]))
        ri_mol.update_to_new_structure_with_delta_v(v_change)
        assert np.allclose(ri_mol.ic_values,
                           np.array([2.01413724, 2.01413724, 1.9106340176]))

    def test_set_new_vspace(self):
        with path('saddle.test.data', 'water.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        ri_mol = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        ri_mol.add_bond(1, 0)
        ri_mol.add_bond(1, 2)
        ri_mol.add_angle(0, 1, 2)
        ri_mol.set_key_ic_number(1)
        new_vp = np.eye(3)
        ri_mol.set_vspace(new_vp)
        assert (np.allclose(ri_mol.vspace, np.eye(3)))

    def test_align_v_space(self):
        with path('saddle.test.data', 'water.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        mol_1 = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        mol_1.add_bond(1, 0)
        mol_1.add_bond(1, 2)
        mol_1.add_angle(0, 1, 2)
        mol_1.set_key_ic_number(1)
        mol_2 = deepcopy(mol_1)
        mol_1.set_target_ic([2., 2., 2.])
        mol_1.converge_to_target_ic()
        assert np.allclose(mol_1.ic_values, [2., 2., 2.], atol=1e-4)
        copy1 = deepcopy(mol_1)
        copy2 = deepcopy(mol_2)
        copy1.align_vspace(copy2)
        assert np.allclose(copy1.vspace, mol_2.vspace)
        with path('saddle.test.data', 'water_1.fchk') as fchk_path:
            # print 'cv2',copy2.vspace
            copy2.energy_from_fchk(fchk_path)
        # print 'cv2, new',copy2.vspace, copy2.vspace_gradient
        ref_ic_gradient = np.dot(copy2.vspace, copy2.vspace_gradient)
        # print 'cv2,energy'
        copy2.align_vspace(copy1)
        # print 'cv2', copy2.vspace, copy2.vspace_gradient
        new_ic_gradient = np.dot(copy2.vspace, copy2.vspace_gradient)
        assert np.allclose(ref_ic_gradient, new_ic_gradient)
        assert np.allclose(copy1.vspace, copy2.vspace)

    def test_select_key_ic(self):
        with path('saddle.test.data', 'water.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        mol_1 = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        mol_1.add_bond(1, 0)
        mol_1.add_bond(1, 2)
        mol_1.add_angle(0, 1, 2)
        mol_1.select_key_ic(0)
        assert mol_1.key_ic_number == 1
        mol_1.select_key_ic(2)
        assert mol_1.key_ic_number == 1
        mol_1.select_key_ic(0, 1, 2)
        assert mol_1.key_ic_number == 3
