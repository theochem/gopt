import os
import unittest
from copy import deepcopy

import numpy as np
from importlib_resources import path
from numpy.testing import assert_allclose
from saddle.errors import InvalidArgumentError
from saddle.internal import Internal
from saddle.path_ri import PathRI
from saddle.reduced_internal import ReducedInternal
from saddle.ts_construct import TSConstruct
from saddle.utils import Utils


class Test_TS_Construct(unittest.TestCase):

    file_list = []

    def setUp(self):
        with path('saddle.test.data', 'ch3_hf.xyz') as rct_path:
            self.rct = Utils.load_file(rct_path)
        with path('saddle.test.data', 'ch3f_h.xyz') as prd_path:
            self.prd = Utils.load_file(prd_path)

        self.reactant_ic = Internal(self.rct.coordinates, self.rct.numbers, 0,
                                    2)
        self.product_ic = Internal(self.prd.coordinates, self.prd.numbers, 0,
                                   2)
    def test_auto_internal(self):
        ts_ins = TSConstruct(self.reactant_ic, self.product_ic)
        assert isinstance(ts_ins.reactant, Internal)
        assert isinstance(ts_ins.product, Internal)
        ts_ins.auto_select_ic()
        assert len(ts_ins.reactant.ic) == len(ts_ins.product.ic)
        self.reactant_ic.auto_select_ic()
        print(ts_ins.reactant.ic)
        print(self.reactant_ic.ic)
        # self.reactant_ic.add_bond(0, 4)
        # self.reactant_ic.add_angle(1, 0, 4)
        # self.reactant_ic.add_angle(2, 0, 4)
        # assert len(self.reactant_ic.ic) != len(ts_ins.reactant.ic)
        # self.reactant_ic.add_angle(3, 0, 4)
        assert len(ts_ins.reactant.ic_values) - len(
            self.reactant_ic.ic_values) == 11

    def test_auto_ic_create(self):
        ts_ins = TSConstruct(self.reactant_ic, self.product_ic)
        ts_ins.auto_select_ic()
        ref_ic_rct = np.array([
            2.02762919, 2.02769736, 2.02761705, 1.77505755, 4.27707385,
            4.87406146, 2.08356856, 2.08391343, 1.64995596, 2.08364916,
            1.64984524, 1.64881837, 1.06512165, 0.427652638, 3.14154596,
            2.71390135, 0.594853893, -1.70630517, 1.70613580, -3.14152957,
            2.09455878, -2.09427619, -2.87079827, 6.05213140, 1.64996828,
            1.64984426, 1.64880703, 1.36936807e-05, 3.29954545e-05,
            -0.452180460, 1.64217144, -2.54673936, 2.68941267, -1.49942229
        ])
        ref_ic_prd = np.array([
            2.03992597, 2.03991419, 2.03976417, 5.52444423, 8.17667938,
            9.06322941, 1.91251903, 1.9119936, 1.92886636, 1.91202283,
            1.88746552, 1.91077332, 1.01701674, 0.2138026, 0.01152089,
            0.2133746, -1.50710794, -2.11774407, 2.06704189, 0.05432083,
            2.10752341, -2.0822473, -2.09807877, 2.6533652, 1.90903961,
            1.90907132, 1.90914538, 0.02398886, 3.1060829, -2.54607887,
            -0.4513039, 1.6429288, 0.60041547, 2.69383007
        ])
        print(ts_ins.reactant.ic_values, 'r')
        print(ts_ins.product.ic_values, 'p')
        assert np.allclose(ts_ins.reactant.ic_values, ref_ic_rct)
        assert np.allclose(ts_ins.product.ic_values, ref_ic_prd)

    def test_ts_construct(self):
        ts_ins = TSConstruct(self.reactant_ic, self.product_ic)
        ts_ins.auto_select_ic()
        ts_ins.create_ts_state(start_with="product")
        result = deepcopy(ts_ins.ts)
        ts_ins.create_ts_state(start_with="reactant")
        result_2 = deepcopy(ts_ins.ts)
        # print result_2.ic_values
        ref_tar_ic = ts_ins._reactant.ic_values * 0.5 + \
            ts_ins._product.ic_values * 0.5
        assert np.allclose(ref_tar_ic, result.target_ic)
        assert np.allclose(result.target_ic, result_2.target_ic)
        # assert np.allclose(
        #     result.ic_values[:4], result.target_ic[:4], atol=1e-6)
        # TODO: need to check structure
        # assert np.allclose(
        #     result_2.ic_values[:4], result_2.target_ic[:4], atol=1e-6)
        assert_allclose(
            result.ic_values[:16], result_2.ic_values[:16], atol=1e-3)
        ts_ins.select_key_ic(1)
        assert ts_ins.key_ic_counter == 1
        assert np.allclose(ts_ins.ts.ic_values[:2][::-1],
                           result_2.ic_values[:2])
        assert np.allclose(ts_ins.ts._cc_to_ic_gradient[1],
                           result_2._cc_to_ic_gradient[0])

    def test_ts_union(self):
        self.reactant_ic.add_bond(0, 1)
        self.reactant_ic.add_bond(1, 2)
        self.reactant_ic.add_angle(0, 1, 2)
        self.product_ic.add_bond(1, 0)
        self.product_ic.add_bond(0, 2)
        self.product_ic.add_angle(1, 0, 2)
        ts_ins = TSConstruct(self.reactant_ic, self.product_ic)
        union_ic = ts_ins._get_union_of_ics()
        assert len(union_ic) == 5

    def test_ts_union_reactant(self):
        self.reactant_ic.add_bond(0, 1)
        self.reactant_ic.add_bond(1, 2)
        self.reactant_ic.add_angle(0, 1, 2)
        ts_ins = TSConstruct(self.reactant_ic, self.product_ic)
        union_ic = ts_ins._get_union_of_ics(mode='reactant')
        assert len(union_ic) == 3
        ts_ins.auto_select_ic(auto_select=False, mode='reactant')
        assert len(ts_ins.product.ic) == 3

    def test_ts_union_product(self):
        self.product_ic.add_bond(1, 0)
        self.product_ic.add_bond(0, 2)
        self.product_ic.add_angle(1, 0, 2)
        ts_ins = TSConstruct(self.reactant_ic, self.product_ic)
        union_ic = ts_ins._get_union_of_ics(mode='product')
        assert len(union_ic) == 3
        ts_ins.auto_select_ic(auto_select=False, mode='product')
        assert len(ts_ins.reactant.ic) == 3
        flag = 1
        try:
            ts_ins.auto_select_ic(auto_select=False, mode='wrong')
        except InvalidArgumentError:
            flag = 0
        assert flag == 0

    def test_ts_create(self):
        ts_ins = TSConstruct(self.reactant_ic, self.product_ic)
        ts_ins.auto_select_ic()
        ts_ins.create_ts_state(start_with='reactant')
        ts_ins.select_key_ic(3, 4)
        assert isinstance(ts_ins.ts.ic[0], type(ts_ins.product.ic[3]))
        assert isinstance(ts_ins.product.ic[3], type(ts_ins.ts.ic[0]))
        assert isinstance(ts_ins.ts.ic[1], type(ts_ins.product.ic[4]))
        assert isinstance(ts_ins.product.ic[4], type(ts_ins.ts.ic[1]))
        assert ts_ins.ts.key_ic_number == 2
        new_ins = TSConstruct(self.reactant_ic, self.product_ic)
        assert new_ins is not ts_ins
        new_ins.auto_generate_ts()
        new_ins.select_key_ic(3, 4)
        assert np.allclose(new_ins.ts.ic_values, ts_ins.ts.ic_values)
        assert isinstance(new_ins.ts, ReducedInternal)

    def test_ts_combine(self):  # maybe a problem
        self.reactant_ic.auto_select_ic()
        self.product_ic.auto_select_ic()
        new_ins = TSConstruct(self.reactant_ic, self.product_ic)
        new_ins.auto_generate_ts(start_with='product', reset_ic=False)
        assert all(
            np.abs(np.dot(new_ins.ts.b_matrix.T, new_ins.ts._cost_q_d)) < 3e-4)
        e_v = np.linalg.eigh(new_ins.ts.cost_value_in_cc[2])[0]
        assert all(e_v[np.abs(e_v) > 1e-4] > 0)

    def test_choices_auto_select_ic(self):
        self.reactant_ic.add_bond(2, 4)
        new_ins = TSConstruct(self.reactant_ic, self.product_ic)
        new_ins.auto_generate_ts(auto_select=False)
        print('rct', self.reactant_ic.ic)
        print('prd', self.product_ic.ic)
        ref_ic = (self.reactant_ic.distance(2, 4) + self.product_ic.distance(
            2, 4)) / 2
        assert np.allclose(new_ins.ts.ic_values, ref_ic)
        new_ins = TSConstruct(self.reactant_ic, self.product_ic)
        print('rct', new_ins.reactant.ic)
        print('prd', new_ins.product.ic)
        new_ins.auto_generate_ts(auto_select=True, reset_ic=False)
        # print('rct', new_ins.reactant.ic)
        # print('prd', new_ins.product.ic)
        print('ts', new_ins.ts.ic)
        print('target_ic', new_ins.ts.target_ic)
        print('ts g', new_ins.ts._compute_tfm_gradient())
        # print('ts g', new_ins.)
        # with deepcopy 31, no deepcopy 44
        assert len(new_ins.ts.ic) == 31
        # TODO: need to be reviewed
        # print(new_ins.ts.ic)
        print(new_ins.ts.tf_cost)
        assert np.allclose(new_ins.ts.ic_values[1:5], new_ins.ts.target_ic[1:5], atol=2e-2)
        new_ins = TSConstruct(self.reactant_ic, self.product_ic)
        new_ins.auto_generate_ts(auto_select=True, reset_ic=True)
        assert all(
            np.abs(np.dot(new_ins.ts.b_matrix.T, new_ins.ts._cost_q_d)) < 3e-4)

    # def test_from_file_and_to_file(self):
    #     with path('saddle.test.data', 'ch3_hf.xyz') as rct_p:
    #         with path('saddle.test.data', 'ch3f_h.xyz') as prd_p:
    #             ts = TSConstruct.from_file(rct_p, prd_p)
    #     ts_ins = TSConstruct(self.reactant_ic, self.product_ic)
    #     ts.auto_generate_ts()
    #     ts_ins.auto_generate_ts()
    #     with path('saddle.test.data', 'ts_nose_test_cons.xyz') as filepath:
    #         ts.ts_to_file(filepath)
    #     self.file_list.append(filepath)
    #     mol = Utils.from_file(filepath)
    #     assert np.allclose(mol.coordinates, ts.ts.coordinates)

    def test_from_file_to_path(self):
        with path('saddle.test.data', 'rct.xyz') as rct_path:
            with path('saddle.test.data', 'prd.xyz') as prd_path:
                ts_mol = TSConstruct.from_file(rct_path, prd_path)
        ts_mol.auto_generate_ts(task='path')
        assert isinstance(ts_mol.ts, PathRI)

    def test_update_rct_prd_structure(self):
        with path('saddle.test.data', 'rct.xyz') as rct_path:
            with path('saddle.test.data', 'prd.xyz') as prd_path:
                ts_mol = TSConstruct.from_file(rct_path, prd_path)
        ts_mol.auto_generate_ts(task='path')
        assert len(ts_mol.ts.ic) == 9
        ts_mol.ts.auto_select_ic(keep_bond=True)
        assert len(ts_mol.ts.ic) == 12
        ts_mol.update_rct_and_prd_with_ts()
        assert len(ts_mol.rct.ic) == 12
        assert len(ts_mol.prd.ic) == 12
        for i in range(12):
            assert ts_mol.rct.ic[i].atoms == ts_mol.prd.ic[i].atoms

    def test_dihed_special_structure(self):
        with path('saddle.test.data', 'rct.xyz') as rct_path:
            with path('saddle.test.data', 'prd.xyz') as prd_path:
                ts_mol = TSConstruct.from_file(rct_path, prd_path)
        ts_mol.auto_generate_ts(dihed_special=True)
        assert len(ts_mol.ts.ic) == 11

    @classmethod
    def tearDownClass(cls):
        for i in cls.file_list:
            os.remove(i)
