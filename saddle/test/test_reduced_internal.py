import os
from copy import deepcopy

import numpy as np

from ..internal import Internal
from ..iodata import IOData
from ..reduced_internal import ReducedInternal


class TestReduceInternal(object):
    @classmethod
    def setup_class(self):
        path = os.path.dirname(os.path.realpath(__file__))
        mol_path = path + "/../data/water.xyz"
        mol = IOData.from_file(mol_path)
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
        mole = deepcopy(self.red_int)
        assert np.allclose(mole.ic_values, [1.81413724, 1.81413724, 2.96247453,
                                            -0.33333407, 0.81649681])
        mole.select_key_ic(0, 2)
        assert np.allclose(mole.ic_values, [1.81413724, 2.96247453, 1.81413724,
                                            -0.33333407, 0.81649681])
        svd = mole._svd_of_cc_to_ic_gradient()
        ref_vectors = np.array(
            [[-4.62909977e-01, 6.15617488e-01, -4.67707997e-01],
             [-7.55929035e-01, 7.65628443e-02, 3.66106995e-01],
             [-4.62909977e-01, -7.40644124e-01, -1.30142388e-01],
             [-1.15470541e-16, -1.60808621e-01, -7.68952115e-01],
             [-6.78941967e-17, 2.01841790e-01, 1.97460125e-01]])
        for i in range(svd.shape[1]):
            assert (np.allclose(svd[:, i], ref_vectors[:, i]) or
                    np.allclose(svd[:, i], -1 * ref_vectors[:, i]))

        ref_unit = np.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.], [0., 0.]])
        assert np.allclose(mole._reduced_unit_vectors(), ref_unit)
        ptrb = mole._reduced_perturbation()
        ref_vec = np.array(
            [[0.81202131, 0.22582935], [0.22582935, 0.71132491],
             [-0.18079919, 0.24557523], [0.26064845, -0.29383071],
             [0.03190366, 0.08774511]])
        assert np.allclose(ref_vec, ptrb)
        ref_space = np.array([[0.4554686, -0.77754078],  # reduced internal
                              [-0.56819685, -0.62327943],
                              [-0.41841169, -0.01257065],
                              [0.53869743, -0.01966288],
                              [-0.06661405, -0.08005273]])
        mole._generate_reduce_space()
        assert np.allclose(mole._red_space, ref_space)
        non_red_vec_ref = np.array([[-5.28668553e-17],  # nonreduced internal
                                    [2.22044605e-16],
                                    [7.77528162e-01],
                                    [5.71458853e-01],
                                    [-2.62459020e-01]])
        mole._generate_nonreduce_space()
        assert (np.allclose(non_red_vec_ref, mole._non_red_space) or
                np.allclose(non_red_vec_ref, -1 * mole._non_red_space))
        vp_ref = np.array(
            [[4.55468597e-01, -7.77540781e-01, -5.28668553e-17],
             [-5.68196853e-01, -6.23279427e-01, 2.22044605e-16],
             [-4.18411690e-01, -1.25706498e-02, 7.77528162e-01],
             [5.38697428e-01, -1.96628811e-02, 5.71458853e-01],
             [-6.66140496e-02, -8.00527323e-02, -2.62459020e-01]])
        mole._reset_v_space()
        for i in range(vp_ref.shape[1]):
            assert (np.allclose(mole.vspace[:, i], vp_ref[:, i]) or
                    np.allclose(mole.vspace[:, i], -1 * vp_ref[:, i]))

    def test_ic_ric_transform(self):
        path = os.path.dirname(os.path.realpath(__file__))
        mol_path = path + "/../data/water.xyz"
        mol = IOData.from_file(mol_path)
        ri_mol = Internal(mol.coordinates, mol.numbers, 0, 1)
        ri_mol.add_bond(1, 0)
        ri_mol.add_bond(1, 2)
        ri_mol.add_bond(0, 2)
        ri_mol.add_angle_cos(0, 1, 2)
        ri_mol.add_angle_cos(1, 0, 2)
        vc_ref = np.array(
            [1.81413724, 1.81413724, 2.96247453, -0.33333407, 0.81649681])
        assert np.allclose(ri_mol.ic_values, vc_ref)
        ri_mol = ReducedInternal.update_to_reduced_internal(ri_mol)
        assert isinstance(ri_mol, ReducedInternal)
        ri_mol.set_key_ic_number(2)
        ri_mol.select_key_ic(0, 2)
        vp_ref = np.array(
            [[4.55468597e-01, -7.77540781e-01, -5.28668553e-17],
             [-5.68196853e-01, -6.23279427e-01, 2.22044605e-16],
             [-4.18411690e-01, -1.25706498e-02, 7.77528162e-01],
             [5.38697428e-01, -1.96628811e-02, 5.71458853e-01],
             [-6.66140496e-02, -8.00527323e-02, -2.62459020e-01]])
        for i in range(vp_ref.shape[1]):
            assert (np.allclose(ri_mol.vspace[:, i], vp_ref[:, i]) or
                    np.allclose(ri_mol.vspace[:, i], -1 * vp_ref[:, i]))
        ri_mol.set_key_ic_number(1)
        assert ri_mol._red_space is None
        assert ri_mol._non_red_space is None
        ri_mol.set_key_ic_number(2)
        for i in range(vp_ref.shape[1]):
            assert (np.allclose(ri_mol.vspace[:, i], vp_ref[:, i]) or
                    np.allclose(ri_mol.vspace[:, i], -1 * vp_ref[:, i]))
        new_coor = np.array([[1.40, -0.93019123, -0.], [-0., 0.11720081, -0.],
                             [-1.40, -0.93019123, -0.]])
        ri_mol.set_new_coordinates(new_coor)
        ref_ic = [1.7484364736491811, 2.8, 1.7484364736491811,
                  -0.28229028459335431, 0.8007154]
        assert np.allclose(ri_mol.ic_values, ref_ic)
        ri_mol.vspace
        assert ri_mol._red_space is not None
        assert ri_mol._non_red_space is not None
        ri_mol.add_angle_cos(0, 2, 1)
        assert ri_mol._red_space is None
        assert ri_mol._non_red_space is None

    def test_ric_add_ic(self):
        path = os.path.dirname(os.path.realpath(__file__))
        mol_path = path + "/../data/water.xyz"
        mol = IOData.from_file(mol_path)
        ri_mol = Internal(mol.coordinates, mol.numbers, 0, 1)
        ri_mol = ReducedInternal.update_to_reduced_internal(ri_mol)
        ri_mol.add_bond(1, 0)
        ri_mol.add_bond(1, 2)
        ri_mol.add_bond(0, 2)
        ri_mol.vspace
        ri_mol.add_angle_cos(0, 1, 2)
        ri_mol.add_angle_cos(1, 0, 2)
        ri_mol.select_key_ic(0, 2)
        vp_ref = np.array(
            [[4.55468597e-01, -7.77540781e-01, -5.28668553e-17],
             [-5.68196853e-01, -6.23279427e-01, 2.22044605e-16],
             [-4.18411690e-01, -1.25706498e-02, 7.77528162e-01],
             [5.38697428e-01, -1.96628811e-02, 5.71458853e-01],
             [-6.66140496e-02, -8.00527323e-02, -2.62459020e-01]])
        for i in range(vp_ref.shape[1]):
            assert (np.allclose(ri_mol.vspace[:, i], vp_ref[:, i]) or
                    np.allclose(ri_mol.vspace[:, i], -1 * vp_ref[:, i]))
        ri_mol.set_key_ic_number(1)
        assert ri_mol._red_space is None
        assert ri_mol._non_red_space is None

    def test_get_delta_v(self):
        path = os.path.dirname(os.path.realpath(__file__))
        mol_path = path + "/../data/water.xyz"
        mol = IOData.from_file(mol_path)
        ri_mol = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        ri_mol.add_bond(1, 0)
        ri_mol.add_bond(1, 2)
        ri_mol.add_angle_cos(0, 1, 2)
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
                           np.array([2.01413724, 2.01413724, -0.33333407]))

    def test_set_new_vspace(self):
        path = os.path.dirname(os.path.realpath(__file__))
        mol_path = path + "/../data/water.xyz"
        mol = IOData.from_file(mol_path)
        ri_mol = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        ri_mol.add_bond(1, 0)
        ri_mol.add_bond(1, 2)
        ri_mol.add_angle_cos(0, 1, 2)
        ri_mol.set_key_ic_number(1)
        new_vp = np.eye(3)
        ri_mol.set_vspace(new_vp)
        assert(np.allclose(ri_mol.vspace, np.eye(3)))

    def test_align_v_space(self):
        path = os.path.dirname(os.path.realpath(__file__))
        mol_path = path + "/../data/water.xyz"
        mol = IOData.from_file(mol_path)
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
        fchk_path = path + "/../data/water_1.fchk"
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

    def test_select_key_ic(self):
        path = os.path.dirname(os.path.realpath(__file__))
        mol_path = path + "/../data/water.xyz"
        mol = IOData.from_file(mol_path)
        mol_1 = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        mol_1.add_bond(1, 0)
        mol_1.add_bond(1, 2)
        mol_1.add_angle_cos(0, 1, 2)
        mol_1.select_key_ic(0)
        assert mol_1.key_ic_number == 1
        mol_1.select_key_ic(2)
        assert mol_1.key_ic_number == 1
        mol_1.select_key_ic(0, 1, 2)
        assert mol_1.key_ic_number == 3
