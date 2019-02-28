import unittest
from copy import deepcopy

import numpy as np
from importlib_resources import path
from saddle.internal import Internal
from saddle.utils import Utils
from saddle.molmod import dihed_angle
from saddle.opt import Point
from saddle.coordinate_types import DihedralAngle


class TestInternal(unittest.TestCase):
    def setUp(self):
        with path('saddle.test.data', 'water.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        self.mol = Internal(mol.coordinates, mol.numbers, 0, 1)

    def test_connectivity(self):
        assert np.allclose(self.mol.connectivity, np.eye(3) * -1)

    def test_file_title(self):
        new_mol = Internal(self.mol.coordinates, self.mol.numbers, 0, 1)
        assert len(new_mol._title) == 15

    def test_add_bond(self):
        init_con = np.eye(3) * -1
        assert np.allclose(self.mol.connectivity, init_con)
        self.mol.add_bond(1, 0)
        new_con = init_con.copy()
        new_con[0, 1] = 1
        new_con[1, 0] = 1
        assert np.allclose(self.mol.connectivity, new_con)
        assert np.allclose(self.mol.ic_values, np.array([1.81413724]))
        assert self.mol.ic[0].atoms == (0, 1)
        self.mol.add_bond(0, 1)
        ref_hessian = np.array(
            [[[
                0.18374187, 0.25985046, 0., -0.18374187, -0.25985046, 0., 0.,
                0., 0.
            ],
              [
                  0.25985046, 0.36748434, 0., -0.25985046, -0.36748434, 0., 0.,
                  0., 0.
              ], [0., 0., 0.55122621, 0., 0., -0.55122621, 0., 0., 0.],
              [
                  -0.18374187, -0.25985046, 0., 0.18374187, 0.25985046, 0., 0.,
                  0., 0.
              ],
              [
                  -0.25985046, -0.36748434, 0., 0.25985046, 0.36748434, 0., 0.,
                  0., 0.
              ], [0., 0., -0.55122621, 0., 0., 0.55122621, 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0.]]])
        assert np.allclose(self.mol._cc_to_ic_hessian, ref_hessian)
        assert len(self.mol.ic) == 1
        assert np.allclose(self.mol.connectivity, new_con)
        self.mol.add_bond(2, 1)
        assert len(self.mol.ic) == 2
        assert self.mol.ic[1].atoms == (1, 2)
        assert np.allclose(
            self.mol._cc_to_ic_gradient,
            np.array([[
                0.81649681, -0.57734995, 0., -0.81649681, 0.57734995, 0., 0.,
                0., 0.
            ],
                      [
                          0., 0., 0., 0.81649681, 0.57734995, 0., -0.81649681,
                          -0.57734995, 0.
                      ]]))

    def test_angle_add(self):
        self.mol.add_angle(0, 1, 2)
        assert len(self.mol.ic) == 0
        self.mol.add_bond(0, 1)
        self.mol.add_bond(1, 2)
        connected_index = self.mol.connected_indices(1)
        assert np.allclose(connected_index, np.array([0, 2]))
        self.mol.add_angle(0, 1, 2)
        assert len(self.mol.ic) == 3
        assert np.allclose(self.mol.ic[2].value, 1.9106340153991836)
        assert np.allclose(
            self.mol.b_matrix[2],
            np.array([
                0.31825043, 0.45007444, 0., 0., -0.90014888, -0., -0.31825043,
                0.45007444, 0.
            ]))
        assert np.allclose(
            self.mol._cc_to_ic_hessian[2],
            np.array([[
                -2.86472766e-01, -1.01283669e-01, 0.00000000e+00,
                2.86472766e-01, 1.01283669e-01, 0.00000000e+00, 2.08166817e-17,
                -6.93889390e-18, 0.00000000e+00
            ],
                      [
                          -1.01283669e-01, 2.86472766e-01, 0.00000000e+00,
                          1.01283669e-01, -2.86472766e-01, 0.00000000e+00,
                          6.93889390e-18, 4.16333634e-17, 0.00000000e+00
                      ],
                      [
                          0.00000000e+00, 0.00000000e+00, -1.07427583e-01,
                          0.00000000e+00, 0.00000000e+00, 4.29709622e-01,
                          0.00000000e+00, 0.00000000e+00, -3.22282039e-01
                      ],
                      [
                          2.86472766e-01, 1.01283669e-01, 0.00000000e+00,
                          -5.72945532e-01, 0.00000000e+00, 0.00000000e+00,
                          2.86472766e-01, -1.01283669e-01, 0.00000000e+00
                      ],
                      [
                          1.01283669e-01, -2.86472766e-01, 0.00000000e+00,
                          0.00000000e+00, 5.72945532e-01, 0.00000000e+00,
                          -1.01283669e-01, -2.86472766e-01, 0.00000000e+00
                      ],
                      [
                          0.00000000e+00, 0.00000000e+00, 4.29709622e-01,
                          0.00000000e+00, 0.00000000e+00, -8.59419245e-01,
                          0.00000000e+00, 0.00000000e+00, 4.29709622e-01
                      ],
                      [
                          2.08166817e-17, 6.93889390e-18, 0.00000000e+00,
                          2.86472766e-01, -1.01283669e-01, 0.00000000e+00,
                          -2.86472766e-01, 1.01283669e-01, 0.00000000e+00
                      ],
                      [
                          -6.93889390e-18, 4.16333634e-17, 0.00000000e+00,
                          -1.01283669e-01, -2.86472766e-01, 0.00000000e+00,
                          1.01283669e-01, 2.86472766e-01, 0.00000000e+00
                      ],
                      [
                          0.00000000e+00, 0.00000000e+00, -3.22282039e-01,
                          0.00000000e+00, 0.00000000e+00, 4.29709622e-01,
                          0.00000000e+00, 0.00000000e+00, -1.07427583e-01
                      ]]))
        self.mol.set_target_ic((1.6, 1.7, -0.5))
        assert np.allclose(self.mol.target_ic, np.array([1.6, 1.7, -0.5]))

    def test_dihedral_add(self):
        with path('saddle.test.data', '2h-azirine.xyz') as mol_path:
            mol = Utils.load_file(mol_path)  # create a water molecule
        internal = Internal(mol.coordinates, mol.numbers, 0, 1)
        internal.add_bond(0, 1)
        internal.add_bond(1, 2)
        internal.add_bond(1, 3)
        # fake add dihed
        internal.add_dihedral(0, 2, 3, 4)
        assert len(internal.ic) == 3
        internal.add_dihedral(0, 1, 2, 3)
        assert len(internal.ic) == 4
        assert internal.ic_values[3] == dihed_angle(internal.coordinates[:4])
        #assert np.allclose(internal._cc_to_ic_hessian[3][:12, :12],
        #                   ref_hessian)

    def test_cost_function(self):
        self.mol.add_bond(0, 1)
        self.mol.add_bond(1, 2)
        self.mol.add_angle(0, 1, 2)
        assert np.allclose(
            self.mol.ic_values,
            [1.8141372422079882, 1.8141372422079882, 1.9106340153991836])
        self.mol.set_target_ic([1.7, 1.7, 1.5])
        self.mol.swap_internal_coordinates(0, 2)
        assert np.allclose(
            self.mol.ic_values,
            [1.9106340153991836, 1.8141372422079882, 1.8141372422079882])
        assert np.allclose(self.mol.target_ic, [1.5, 1.7, 1.7])
        self.mol.swap_internal_coordinates(0, 2)
        assert np.allclose(
            self.mol.ic_values,
            [1.8141372422079882, 1.8141372422079882, 1.9106340153991836])
        assert np.allclose(self.mol.target_ic, [1.7, 1.7, 1.5])
        # test cost function in ic
        v = self.mol._cost_v
        d = self.mol._cost_q_d
        dd = self.mol._cost_q_dd

        # calculate ref cost value
        ref_cost = (self.mol.ic_values[0] - 1.7)**2 * 2 + (
            np.cos(self.mol.ic_values[-1]) - np.cos(1.5))**2
        assert np.allclose(ref_cost, v)
        ref_gradient = np.array(
            [0.22827448441597653, 0.22827448441597653, 0.76192388])
        assert np.allclose(d, ref_gradient)
        ref_hessian = np.array([[
            2.,
            0.,
            0.,
        ], [
            0.,
            2.,
            0.,
        ], [
            0.,
            0.,
            1.5083953582,
        ]])
        assert np.allclose(dd, ref_hessian)
        # assert False
        new_v, xd, xdd = self.mol.cost_value_in_cc
        assert new_v == v
        assert np.allclose(xd, np.dot(self.mol._cc_to_ic_gradient.T, d))
        ref_x_hessian = np.dot(
            np.dot(self.mol._cc_to_ic_gradient.T, dd),
            self.mol._cc_to_ic_gradient)
        K = np.tensordot(d, self.mol._cc_to_ic_hessian, 1)
        ref_x_hessian += K
        assert np.allclose(xdd, ref_x_hessian)
        new_coor = np.array([[1.40, -0.93019123, -0.], [-0., 0.11720081, -0.],
                             [-1.40, -0.93019123, -0.]])
        self.mol.set_new_coordinates(new_coor)
        assert np.allclose(
            self.mol.ic_values,
            [1.7484364736491811, 1.7484364736491811, 1.85697699])
        assert np.allclose(
            self.mol._cc_to_ic_gradient[0, :6],
            np.array(
                [0.80071539, -0.59904495, 0., -0.80071539, 0.59904495, -0.]))
        ref_hessian = np.array(
            [[0.20524329, 0.27433912, 0., -0.20524329, -0.27433912, -0.],
             [0.27433912, 0.36669628, 0., -0.27433912, -0.36669628, -0.],
             [0., 0., 0.57193957, -0., -0., -0.57193957],
             [-0.20524329, -0.27433912, -0., 0.20524329, 0.27433912, 0.],
             [-0.27433912, -0.36669628, -0., 0.27433912, 0.36669628, 0.],
             [-0., -0., -0.57193957, 0., 0., 0.57193957]])
        assert np.allclose(self.mol._cc_to_ic_hessian[0, :6, :6], ref_hessian)

    def test_transform_function(self):
        self.mol.add_bond(0, 1)
        self.mol.add_bond(1, 2)
        self.mol.add_angle(0, 1, 2)
        self.mol.set_target_ic([1.7, 1.7, -0.4])
        n_p = self.mol.create_geo_point()
        assert isinstance(n_p, Point)
        assert n_p.trust_radius == 1.7320508075688772

        self.mol.converge_to_target_ic(iteration=100)
        g_array = self.mol.cost_value_in_cc[1]
        assert len(g_array[abs(g_array) > 3e-4]) == 0

    def test_auto_ic_select_water(self):
        self.mol.auto_select_ic()
        assert np.allclose(
            self.mol.ic_values,
            [1.8141372422079882, 1.8141372422079882, 1.9106340153991836])

    def test_auto_ic_select_ethane(self):
        with path('saddle.test.data', 'ethane.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        ethane = Internal(mol.coordinates, mol.numbers, 0, 1)
        ethane.auto_select_ic()
        assert len(ethane.ic) == 24

    def test_auto_dihed_number_ethane(self):
        with path('saddle.test.data', 'ethane.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        mol = Utils.load_file(mol_path)
        ethane = Internal(mol.coordinates, mol.numbers, 0, 1)
        ethane.auto_select_ic()
        counter = 0
        for ic in ethane.ic:
            if isinstance(ic, DihedralAngle):
                counter += 1
        assert counter == 5

    def test_auto_select_improper_ch3_hf(self):
        with path('saddle.test.data', 'ch3_hf.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        mol = Internal(mol.coordinates, mol.numbers, 0, 1)
        mol.auto_select_ic()
        ic_ref = np.array([
            2.02762919, 2.02769736, 2.02761705, 1.77505755, 4.27707385,
            4.87406146, 2.08356856, 2.08391343, 1.64995596, 2.08364916,
            1.64984524, 1.64881837, 1.06512165, 0.42765264, 3.14154596,
            2.71390135, 0.59485389, -1.70630517, 1.7061358, -3.14152957,
            2.09455878, -2.09427619, -2.87079827
        ])
        assert np.allclose(mol.ic_values, ic_ref)

    def test_auto_ic_select_methanol(self):
        with path('saddle.test.data', 'methanol.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        methanol = Internal(mol.coordinates, mol.numbers, 0, 1)
        methanol.auto_select_ic()
        assert len(methanol.ic) == 15

    def test_wipe_ic(self):
        self.mol.add_bond(0, 1)
        self.mol.add_bond(1, 2)
        self.mol.add_bond(0, 2)
        assert len(self.mol.ic) == 3
        self.mol.wipe_ic_info(False)
        assert len(self.mol.ic) == 3
        self.mol.wipe_ic_info(True)
        assert len(self.mol.ic) == 0

    def test_set_new_ic(self):
        mol1 = deepcopy(self.mol)
        mol2 = deepcopy(self.mol)
        mol1.add_bond(0, 1)
        mol1.add_bond(0, 2)
        mol1.add_angle(1, 0, 2)
        mol2.set_new_ics(mol1.ic)
        assert np.allclose(mol2.ic_values, mol1.ic_values)
        mol2.wipe_ic_info(True)
        mol2.add_bond(2, 0)
        assert len(mol2.ic) == 1
        assert not np.allclose(mol2.connectivity, mol1.connectivity)
        mol2.set_new_ics(mol1.ic)
        assert np.allclose(mol1.connectivity, mol2.connectivity)

    def test_get_energy_from_fchk(self):
        with path('saddle.test.data', 'water_1.fchk') as fchk_path:
            self.mol.add_bond(0, 1)
            self.mol.add_bond(1, 2)
            self.mol.add_angle(0, 1, 2)
            self.mol.energy_from_fchk(fchk_path)
        ref_g = self.mol.internal_gradient.copy()
        assert np.allclose(self.mol.internal_gradient, ref_g)
        ic_ref = deepcopy(self.mol.ic)
        self.mol.add_bond(0, 2)
        assert not np.allclose(self.mol.internal_gradient.shape, ref_g.shape)
        self.mol.set_new_ics(ic_ref)
        assert np.allclose(self.mol.internal_gradient, ref_g)
        self.mol.swap_internal_coordinates(0, 2)
        assert np.allclose(self.mol.internal_gradient[2], ref_g[0])
        assert np.allclose(self.mol.internal_gradient[0], ref_g[2])

    def test_delete_ic(self):
        with path('saddle.test.data', 'ethane.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        ethane = Internal(mol.coordinates, mol.numbers, 0, 1)
        ethane.auto_select_ic()
        ethane._delete_ic_index(0)
        assert len(ethane.ic) == 23
        ethane.auto_select_ic(keep_bond=True)
        assert len(ethane.ic) == 12
        # print(ethane.ic)
        ethane.delete_ic(1, 2, 3)
        assert len(ethane.ic) == 9
        # print(ethane.ic)
        # assert False

    def test_fragments_in_mole(self):
        with path('saddle.test.data', 'ch3_hf.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        mol = Internal(mol.coordinates, mol.numbers, 0, 1)
        assert len(mol.fragments) == mol.natom
        mol.add_bond(0, 1)
        mol.add_bond(2, 3)
        # print(mol.fragments)
        assert len(mol.fragments) == mol.natom - 2
        mol.add_bond(0, 2)
        assert len(mol.fragments) == mol.natom - 3
        mol.add_bond(0, 3)
        assert len(mol.fragments) == mol.natom - 3
        mol.add_bond(4, 5)
        assert len(mol.fragments) == mol.natom - 4
        mol.add_bond(0, 5, b_type=3)
        assert len(mol.fragments) == 2

    def test_fragments_bond_add(self):
        with path('saddle.test.data', 'ch3_hf.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        mol = Internal(mol.coordinates, mol.numbers, 0, 1)
        mol._auto_select_fragment_bond()
        assert len(mol.ic) == 15

        mol.wipe_ic_info(True)
        mol.add_bond(0, 1)
        mol._auto_select_fragment_bond()
        assert len(mol.ic) == 15

        mol.wipe_ic_info(True)
        mol.add_bond(0, 1)
        mol.add_bond(0, 5)
        mol.add_bond(0, 3)
        mol.add_bond(4, 2)
        mol._auto_select_fragment_bond()
        assert len(mol.ic) == 6

        mol.wipe_ic_info(True)
        mol.add_bond(0, 1)
        mol.add_bond(2, 3)
        mol.add_bond(4, 5)
        mol._auto_select_fragment_bond()
        assert len(mol.ic) == 9
        #
        mol.wipe_ic_info(True)
        mol.add_bond(0, 1)
        mol.add_bond(0, 2)
        mol.add_bond(3, 4)
        mol.add_bond(4, 5)
        mol._auto_select_fragment_bond()
        assert np.allclose(mol.ic_values[4], 2.02761704779693)
        assert mol.ic[4].atoms == (0, 3)
        assert np.allclose(mol.ic_values[5], 3.501060110109399)
        assert mol.ic[5].atoms == (2, 3)
        assert len(mol.ic) == 6

    def test_dihedral_rotation(self):
        with path('saddle.test.data', 'h2o2.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        h2o2 = Internal(mol.coordinates, mol.numbers, 0, 1)
        h2o2.auto_select_ic()
        ref_ic = np.array([
            2.47617635, 1.85058569, 1.85070922, 1.81937566, 1.81930967,
            1.43966113
        ])
        assert np.allclose(h2o2.ic_values, ref_ic)
        target_ic = [2.4, 1.8, 1.8, 1.6, 1.6, 1.57]
        h2o2.set_target_ic(target_ic)
        h2o2.converge_to_target_ic()
        assert np.allclose(h2o2.ic_values, target_ic, atol=1e-3)

        target_ic = [2.4, 1.8, 1.8, 1.6, 1.6, 3.14]
        h2o2.set_target_ic(target_ic)
        h2o2.converge_to_target_ic()
        assert np.allclose(h2o2.ic_values, target_ic, atol=1e-4)

        target_ic = [2.4, 1.8, 1.8, 1.6, 1.6, -1.57]
        h2o2.set_target_ic(target_ic)
        h2o2.converge_to_target_ic()
        assert np.allclose(h2o2.ic_values, target_ic, atol=1e-3)

    def test_dihedral_repeak(self):
        with path('saddle.test.data', 'h2o2.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        h2o2 = Internal(mol.coordinates, mol.numbers, 0, 1)
        h2o2.add_bond(0, 1)
        h2o2.add_bond(1, 2)
        h2o2.add_bond(2, 3)
        h2o2.add_bond(3, 2)
        h2o2.add_bond(0, 2)
        h2o2.add_bond(1, 3)
        assert len(h2o2.ic) == 5
        h2o2.add_dihedral(0, 1, 2, 3)
        assert len(h2o2.ic) == 6
        h2o2.add_dihedral(0, 2, 1, 3)
        assert len(h2o2.ic) == 6

    def test_new_dihed(self):
        with path('saddle.test.data', 'h2o2.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        h2o2 = Internal(mol.coordinates, mol.numbers, 0, 1)
        h2o2.add_bond(0, 1)
        h2o2.add_bond(1, 2)
        h2o2.add_bond(2, 3)
        assert len(h2o2.ic) == 3
        h2o2.add_dihedral(0, 1, 2, 3, special=True)
        assert len(h2o2.ic) == 5
        assert h2o2.b_matrix.shape == (5, 12)
        h2o2.add_dihedral(3, 1, 2, 0)
        assert len(h2o2.ic) == 5
        h2o2.add_dihedral(3, 2, 1, 0, special=True)
        assert len(h2o2.ic) == 5
        ref_b = h2o2.b_matrix.copy()
        h2o2._regenerate_ic()
        assert np.allclose(h2o2.b_matrix, ref_b)

    def test_new_dihed_converge(self):
        with path('saddle.test.data', 'h2o2.xyz') as mol_path:
            mol = Utils.load_file(mol_path)
        mol = Utils.load_file(mol_path)
        h2o2 = Internal(mol.coordinates, mol.numbers, 0, 1)
        h2o2.auto_select_ic(dihed_special=True)
        assert len(h2o2.ic) == 7
        print(h2o2.ic_values)
        target_ic = [2.4, 1.8, 1.8, 1.6, 1.6, 0.8, 0.6]
        h2o2.set_target_ic(target_ic)
        h2o2.converge_to_target_ic()
        print(h2o2.ic_values)
        assert np.allclose(h2o2.ic_values, target_ic, atol=1e-2)
        # print(h2o2.ic_values)

    def test_bond_type(self):
        with path('saddle.test.data', 'ch3_hf.xyz') as mol_path:
            mol = Internal.from_file(mol_path, charge=0, multi=1)
        mol._auto_select_cov_bond()  # numbers [6 1 1 1 9 1]
        assert np.sum(mol.connectivity[0] == 1) == 3
        assert np.sum(mol.connectivity[4] == 1) == 1
        mol._auto_select_fragment_bond()
        assert np.sum(mol.connectivity[0] == 1) == 3
        assert len(np.unique(mol.fragments)) == 1
        assert np.sum(mol.connectivity[5] == 3) == 2
        mol._regenerate_ic()
        assert np.sum(mol.connectivity[0] == 1) == 3
        assert len(np.unique(mol.fragments)) == 1
        assert np.sum(mol.connectivity[5] == 3) == 2
        mol.wipe_ic_info(True)
        mol.auto_select_ic()
        assert np.sum(mol.connectivity[0] == 1) == 3
        assert len(np.unique(mol.fragments)) == 1
        assert np.sum(mol.connectivity[5] == 3) == 2

    def test_h_bond(self):
        with path('saddle.test.data', 'di_water.xyz') as mol_path:
            mol = Internal.from_file(mol_path, charge=0, multi=1)
        mol._auto_select_cov_bond()  # numbers [8 1 1 8 1 1]
        assert np.sum(mol.connectivity[0] == 1) == 2
        assert np.sum(mol.connectivity[3] == 1) == 2
        assert len(mol.fragments) == 2
        print(mol.connectivity)
        mol._auto_select_h_bond()
        assert mol.connectivity[2][3] == 2
        assert len(mol.fragments) == 2

    def test_mini_dihed(self):
        with path('saddle.test.data', 'methanol.xyz') as mol_path:
            mol = Internal.from_file(mol_path)
        mol._auto_select_cov_bond()
        mol._auto_select_h_bond()
        mol._auto_select_fragment_bond()
        mol._auto_select_angle()
        # start real parts
        ref = len(mol.ic)
        mol._auto_select_minimum_dihed_normal()
        assert len(mol.ic) - ref == 1

        with path('saddle.test.data', 'ethane.xyz') as mol2_path:
            mol2 = Internal.from_file(mol2_path)
        mol2._auto_select_cov_bond()
        mol2._auto_select_h_bond()
        mol2._auto_select_fragment_bond()
        mol2._auto_select_angle()
        # start real parts
        ref = len(mol2.ic)
        mol2._auto_select_minimum_dihed_normal()
        assert len(mol2.ic) - ref == 1

    def test_b_matrix(self):
        with path('saddle.test.data', 'h2o2.xyz') as mol_path:
            mol = Internal.from_file(mol_path)
        mol.add_bond(0, 1)
        mol.add_bond(1, 2)
        mol.add_angle(0, 1, 2)
        ref_mol = deepcopy(mol)
        q1 = mol.ic_values[0]
        for i in range(3):
            coor = mol.coordinates.copy()
            coor[0][i] += 1e-4
            ref_mol.set_new_coordinates(coor)
            q2 = ref_mol.ic_values[0]
            fd = (q2 - q1) / 1e-4
            b = mol.b_matrix[0][i]
            assert np.allclose(fd, b, atol=1e-4)

        q1 = mol.ic_values[1]
        for i in range(3):
            coor = mol.coordinates.copy()
            coor[1][i] += 1e-4
            ref_mol.set_new_coordinates(coor)
            q2 = ref_mol.ic_values[1]
            fd = (q2 - q1) / 1e-4
            b = mol.b_matrix[1][3 + i]
            assert np.allclose(fd, b, atol=1e-4)

        q1 = mol.ic_values[2]
        for i in range(3):
            coor = mol.coordinates.copy()
            coor[2][i] += 1e-4
            ref_mol.set_new_coordinates(coor)
            q2 = ref_mol.ic_values[2]
            fd = (q2 - q1) / 1e-4
            b = mol.b_matrix[2][6 + i]
            assert np.allclose(fd, b, atol=1e-4)

    def test_b_matrix_dihed(self):
        with path('saddle.test.data', 'h2o2.xyz') as mol_path:
            mol = Internal.from_file(mol_path)
        mol.auto_select_ic()
        ref_mol = deepcopy(mol)
        for j in range(4):
            q1 = mol.ic_values[j]
            for i in range(3):
                coor = mol.coordinates.copy()
                coor[j][i] += 1e-4
                ref_mol.set_new_coordinates(coor)
                q2 = ref_mol.ic_values[j]
                fd = (q2 - q1) / 1e-4
                b = mol.b_matrix[j][j * 3 + i]
                assert np.allclose(fd, b, atol=1e-4)

    def test_tfm_hessian(self):
        with path('saddle.test.data', 'h2o2.xyz') as mol_path:
            mol = Internal.from_file(mol_path)
        mol.add_bond(0, 1)
        mol.add_bond(1, 2)
        mol.add_angle(0, 1, 2)
        ref_mol = deepcopy(mol)
        qd1 = mol.b_matrix
        for i in range(3):
            coor = mol.coordinates.copy()
            coor[0][i] += 1e-4
            ref_mol.set_new_coordinates(coor)
            qd2 = ref_mol.b_matrix
            fd_b = (qd2 - qd1) / 1e-4
            analytic_b = mol._cc_to_ic_hessian[:, :, i]
            assert np.allclose(fd_b, analytic_b, atol=1e-4)

    def test_tfm_hessian_dihed(self):
        with path('saddle.test.data', 'h2o2.xyz') as mol_path:
            mol = Internal.from_file(mol_path)
        mol.auto_select_ic()
        ref_mol = deepcopy(mol)
        qd1 = mol.b_matrix
        for j in range(4):
            for i in range(3):
                coor = mol.coordinates.copy()
                coor[j][i] += 1e-4
                ref_mol.set_new_coordinates(coor)
                qd2 = ref_mol.b_matrix
                fd_b = (qd2 - qd1) / 1e-4
                analytic_b = mol._cc_to_ic_hessian[:, :, 3 * j + i]
                assert np.allclose(fd_b, analytic_b, atol=1e-4)

    def test_cost_tfm_bond(self):
        with path('saddle.test.data', 'h2o2.xyz') as mol_path:
            mol = Internal.from_file(mol_path)
        mol.add_bond(0, 1)
        ref_mol = deepcopy(mol)
        coor = mol.coordinates.copy()
        target_ic = mol.ic_values
        # print(mol.ic)
        target_ic[0] = 2
        mol.set_target_ic(target_ic)
        # print(mol.ic[0].value - mol.ic[0].target)
        cost_v = mol._compute_tfm_cost()
        cost_g = mol._compute_tfm_gradient()
        diff = 1e-4
        assert np.allclose(cost_v, (2 - 2.47617635) ** 2)
        # finite diff test
        for i in range(4):
            for j in range(3):
                coor = ref_mol.coordinates.copy()
                coor[i][j] += diff
                mol.set_new_coordinates(coor)
                cost_v_2 = mol._compute_tfm_cost()
                fd = (cost_v_2 - cost_v) / 1e-4
                print(fd, cost_g[i * 3 + j], i * 3 + j)
                assert np.allclose(fd, cost_g[3 * i + j], atol=2e-4)

        # tests with bond and angle
    def test_cost_tfm_angle(self):
        with path('saddle.test.data', 'h2o2.xyz') as mol_path:
            mol = Internal.from_file(mol_path)
        mol.add_bond(0, 1)
        mol.add_bond(0, 2)
        mol.add_angle(1, 0, 2)
        ref_mol = deepcopy(mol)
        coor = mol.coordinates.copy()
        target_ic = mol.ic_values
        # print(mol.ic)
        target_ic[2] = 2
        target_ic[0] = 2
        mol.set_target_ic(target_ic)
        # print(mol.ic[0].value - mol.ic[0].target)
        cost_v = mol._compute_tfm_cost()
        cost_g = mol._compute_tfm_gradient()
        diff = 1e-4
        # assert np.allclose(cost_v, (2 - 2.47617635) ** 2)
        # finite diff test
        for i in range(4):
            for j in range(3):
                coor = ref_mol.coordinates.copy()
                coor[i][j] += diff
                mol.set_new_coordinates(coor)
                cost_v_2 = mol._compute_tfm_cost()
                fd = (cost_v_2 - cost_v) / 1e-4
                print(fd, cost_g[i * 3 + j], i * 3 + j)
                assert np.allclose(fd, cost_g[3 * i + j], atol=2e-4)

    def test_cost_tfm_dihed(self):
        with path('saddle.test.data', 'h2o2.xyz') as mol_path:
            mol = Internal.from_file(mol_path)
        mol.auto_select_ic()
        ref_mol = deepcopy(mol)
        coor = mol.coordinates.copy()
        target_ic = mol.ic_values
        # print(mol.ic)
        target_ic[0] = 2
        target_ic[3] = 2
        target_ic[5] = 2
        mol.set_target_ic(target_ic)
        # print(mol.ic[0].value - mol.ic[0].target)
        cost_v = mol._compute_tfm_cost()
        cost_g = mol._compute_tfm_gradient()
        diff = 1e-4
        # assert np.allclose(cost_v, (2 - 2.47617635) ** 2)
        # finite diff test
        for i in range(4):
            for j in range(3):
                coor = ref_mol.coordinates.copy()
                coor[i][j] += diff
                mol.set_new_coordinates(coor)
                cost_v_2 = mol._compute_tfm_cost()
                fd = (cost_v_2 - cost_v) / 1e-4
                assert np.allclose(fd, cost_g[3 * i + j], atol=2e-4)

    def test_cost_tfm_dihed_cmplx(self):
        with path('saddle.test.data', 'ethane.xyz') as mol_path:
            mol = Internal.from_file(mol_path)
        mol.auto_select_ic()
        ref_mol = deepcopy(mol)
        coor = mol.coordinates.copy()
        target_ic = mol.ic_values + 0.5
        # print(mol.ic)
        mol.set_target_ic(target_ic)
        # print(mol.ic[0].value - mol.ic[0].target)
        cost_v = mol._compute_tfm_cost()
        cost_g = mol._compute_tfm_gradient()
        diff = 1e-4
        # assert np.allclose(cost_v, (2 - 2.47617635) ** 2)
        # finite diff test
        for i in range(8):
            for j in range(3):
                coor = ref_mol.coordinates.copy()
                coor[i][j] += diff
                mol.set_new_coordinates(coor)
                cost_v_2 = mol._compute_tfm_cost()
                fd = (cost_v_2 - cost_v) / 1e-4
                assert np.allclose(fd, cost_g[3 * i + j], atol=4e-4)

    def test_scipy_opt_tfm(self):
        with path('saddle.test.data', 'h2o2.xyz') as mol_path:
            mol = Internal.from_file(mol_path)
        mol.auto_select_ic()
        target_ic = mol.ic_values
        target_ic[0] = 2
        target_ic[3] = 2
        target_ic[5] = 2
        mol.set_target_ic(target_ic)
        mol.optimize_to_target_ic()
        np.allclose(mol.ic_values[0], 2, atol=1e-4)
        np.allclose(mol.ic_values[3], 2, atol=1e-4)
        np.allclose(mol.ic_values[5], 2, atol=1e-4)

    def test_scipy_opt_tfm_cmpx(self):
        with path('saddle.test.data', 'ethane.xyz') as mol_path:
            mol = Internal.from_file(mol_path)
        mol.auto_select_ic()
        target_ic = mol.ic_values
        target_ic[-1] = -1
        mol.set_target_ic(target_ic)
        print(mol.ic_values)
        mol.optimize_to_target_ic()
        assert np.max(np.abs(mol._compute_tfm_gradient())) < 1e-4
