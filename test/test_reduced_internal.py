from copy import deepcopy

import os

import numpy as np

import horton as ht
from saddle.internal import Internal
from saddle.reduced_internal import ReducedInternal


class TestReduceInternal(object):
    @classmethod
    def setup_class(self):
        fn_xyz = ht.context.get_fn('test/water.xyz')
        mol = ht.IOData.from_file(fn_xyz)
        self.red_int = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        self.red_int.add_bond(1, 0)
        self.red_int.add_bond(1, 2)
        self.red_int.add_bond(0, 2)
        self.red_int.add_angle_cos(0, 1, 2)
        self.red_int.add_angle_cos(1, 0, 2)
        self.red_int.set_key_ic_number(2)

    def test_property(self):
        assert self.red_int.key_ic_number == 2
        assert len(self.red_int.ic) == 5
        assert isinstance(self.red_int, Internal)
        assert self.red_int.df == 3

    def test_reduce_coordinates(self):
        assert np.allclose(self.red_int.ic_values,
                           [1.81413724, 1.81413724, 2.96247453, -0.33333407,
                            0.81649681])
        svd = self.red_int._svd_of_cc_to_ic_gradient()
        ref_vectors = np.array([
            [-4.62909977e-01, 6.15617488e-01, 4.67707997e-01],
            [-4.62909977e-01, -7.40644124e-01, 1.30142388e-01],
            [-7.55929035e-01, 7.65628443e-02, -3.66106995e-01],
            [2.77555756e-16, -1.60808621e-01, 7.68952115e-01],
            [-1.97758476e-16, 2.01841790e-01, -1.97460125e-01]
        ])
        assert np.allclose(svd, ref_vectors)

        ref_unit = np.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.], [0., 0.]])
        assert np.allclose(self.red_int._reduced_unit_vectors(), ref_unit)
        ptrb = self.red_int._reduced_perturbation()
        ref_vec = np.array(
            [[0.81202131, -0.18079919], [-0.18079919, 0.77977641],
             [0.22582935, 0.24557523], [0.26064845, 0.21917522],
             [0.03190366, -0.17519087]])
        assert np.allclose(ref_vec, ptrb)
        ref_space = np.array([[0.52906158, -0.72946223],  # reduced_internal
                              [0.57833907, 0.66730827],
                              [0.4256363, -0.00087946],
                              [0.43076877, -0.04488957],
                              [-0.13744005, -0.14341785]])
        self.red_int._generate_reduce_space()
        assert np.allclose(ref_space, self.red_int._red_space)
        non_red_vec_ref = np.array([[6.88757214e-17],  # non_reduced_internal
                                    [-2.22044605e-16],
                                    [7.28119408e-01],
                                    [-6.55415872e-01],
                                    [2.00679252e-01]])
        self.red_int._generate_nonreduce_space()
        assert np.allclose(non_red_vec_ref, self.red_int._non_red_space)
        vp_ref = np.array([[5.29061584e-01, -7.29462233e-01, 6.88757214e-17],
                           [5.78339073e-01, 6.67308267e-01, -2.22044605e-16],
                           [4.25636302e-01, -8.79460122e-04, 7.28119408e-01],
                           [4.30768766e-01, -4.48895735e-02, -6.55415872e-01],
                           [-1.37440048e-01, -1.43417851e-01, 2.00679252e-01]])
        self.red_int._reset_v_space()
        assert np.allclose(self.red_int.vspace, vp_ref)

    def test_ic_ric_transform(self):
        fn_xyz = ht.context.get_fn('test/water.xyz')
        mol = ht.IOData.from_file(fn_xyz)
        ri_mol = Internal(mol.coordinates, mol.numbers, 0, 1)
        ri_mol.add_bond(1, 0)
        ri_mol.add_bond(1, 2)
        ri_mol.add_bond(0, 2)
        ri_mol.add_angle_cos(0, 1, 2)
        ri_mol.add_angle_cos(1, 0, 2)
        assert np.allclose(ri_mol.ic_values, self.red_int.ic_values)
        ReducedInternal.update_to_reduced_internal(ri_mol)
        assert isinstance(ri_mol, ReducedInternal)
        ri_mol.set_key_ic_number(2)
        vp_ref = np.array([[5.29061584e-01, -7.29462233e-01, 6.88757214e-17],
                           [5.78339073e-01, 6.67308267e-01, -2.22044605e-16],
                           [4.25636302e-01, -8.79460122e-04, 7.28119408e-01],
                           [4.30768766e-01, -4.48895735e-02, -6.55415872e-01],
                           [-1.37440048e-01, -1.43417851e-01, 2.00679252e-01]])
        assert np.allclose(ri_mol.vspace, vp_ref)
        ri_mol.set_key_ic_number(1)
        assert ri_mol._red_space is None
        assert ri_mol._non_red_space is None
        ri_mol.set_key_ic_number(2)
        assert np.allclose(ri_mol.vspace, vp_ref)
        new_coor = np.array([[1.40, -0.93019123, -0.], [-0., 0.11720081, -0.],
                             [-1.40, -0.93019123, -0.]])
        ri_mol.set_new_coordinates(new_coor)
        ref_ic = [1.7484364736491811, 1.7484364736491811, 2.8,
                  -0.28229028459335431, 0.8007154]
        assert np.allclose(ri_mol.ic_values, ref_ic)
        ri_mol.vspace
        assert ri_mol._red_space is not None
        assert ri_mol._non_red_space is not None
        ri_mol.add_angle_cos(0, 2, 1)
        assert ri_mol._red_space is None
        assert ri_mol._non_red_space is None

    def test_ric_add_ic(self):
        fn_xyz = ht.context.get_fn('test/water.xyz')
        mol = ht.IOData.from_file(fn_xyz)
        ri_mol = Internal(mol.coordinates, mol.numbers, 0, 1)
        ReducedInternal.update_to_reduced_internal(ri_mol)
        ri_mol.add_bond(1, 0)
        ri_mol.add_bond(1, 2)
        ri_mol.add_bond(0, 2)
        ri_mol.vspace
        ri_mol.add_angle_cos(0, 1, 2)
        ri_mol.add_angle_cos(1, 0, 2)
        ri_mol.set_key_ic_number(2)
        vp_ref = np.array([[5.29061584e-01, -7.29462233e-01, 6.88757214e-17],
                           [5.78339073e-01, 6.67308267e-01, -2.22044605e-16],
                           [4.25636302e-01, -8.79460122e-04, 7.28119408e-01],
                           [4.30768766e-01, -4.48895735e-02, -6.55415872e-01],
                           [-1.37440048e-01, -1.43417851e-01, 2.00679252e-01]])
        assert np.allclose(ri_mol.vspace, vp_ref)
        ri_mol.set_key_ic_number(1)
        assert ri_mol._red_space is None
        assert ri_mol._non_red_space is None

    def test_get_delta_v(self):
        fn_xyz = ht.context.get_fn('test/water.xyz')
        mol = ht.IOData.from_file(fn_xyz)
        ri_mol = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        ri_mol.add_bond(1, 0)
        ri_mol.add_bond(1, 2)
        ri_mol.add_angle_cos(0, 1, 2)
        v_change = np.array([-0.14714498, 0.12726298, -0.20531072])
        ic_change = np.dot(ri_mol.vspace, v_change)
        assert np.allclose(ic_change, np.array([0.2, 0.2, 0.]))
        ri_mol.update_to_new_structure_with_delta_v(v_change)
        assert np.allclose(ri_mol.ic_values,
                           np.array([2.01413724, 2.01413724, -0.33333407]))

    def test_align_v_space(self):
        fn_xyz = ht.context.get_fn('test/water.xyz')
        mol = ht.IOData.from_file(fn_xyz)
        mol_1 = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        mol_1.add_bond(1, 0)
        mol_1.add_bond(1, 2)
        mol_1.add_angle_cos(0, 1, 2)
        mol_1.set_key_ic_number(1)
        mol_2 = deepcopy(mol_1)
        mol_1.set_target_ic([2.0, 2.0, -0.5])
        mol_1.converge_to_target_ic()
        assert np.allclose(mol_1.ic_values, [2., 2., -0.5])
        copy1 = deepcopy(mol_1)
        copy2 = deepcopy(mol_2)
        copy1.align_vspace(copy2)
        assert np.allclose(copy1.vspace, mol_2.vspace)
        path = os.path.dirname(os.path.realpath(__file__))
        fchk_path = path+"/water_1.fchk"
        #print 'cv2',copy2.vspace
        copy2.energy_from_fchk(fchk_path)
        #print 'cv2, new',copy2.vspace, copy2.vspace_gradient
        ref_ic_gradient = np.dot(copy2.vspace, copy2.vspace_gradient)
        #print 'cv2,energy'
        copy2.align_vspace(copy1)
        #print 'cv2', copy2.vspace, copy2.vspace_gradient
        new_ic_gradient = np.dot(copy2.vspace, copy2.vspace_gradient)
        assert np.allclose(ref_ic_gradient, new_ic_gradient)
        assert np.allclose(copy1.vspace, copy2.vspace)
