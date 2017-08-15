import os
from copy import deepcopy

import numpy as np

from saddle.conf import data_dir
from saddle.errors import InvalidArgumentError
from saddle.internal import Internal
from saddle.iodata import IOData
from saddle.path_ri import Path_RI
from saddle.reduced_internal import ReducedInternal
from saddle.ts_construct import TSConstruct


class Test_TS_Construct(object):

    file_list = []

    def setUp(self):
        self.rct = IOData.from_file(os.path.join(data_dir, "ch3_hf.xyz"))
        self.prd = IOData.from_file(os.path.join(data_dir, "ch3f_h.xyz"))
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
        self.reactant_ic.add_bond(0, 4)
        self.reactant_ic.add_angle_cos(1, 0, 4)
        self.reactant_ic.add_angle_cos(2, 0, 4)
        assert len(self.reactant_ic.ic) != len(ts_ins.reactant.ic)
        self.reactant_ic.add_angle_cos(3, 0, 4)
        assert np.allclose(self.reactant_ic.ic_values,
                           ts_ins.reactant.ic_values)

    def test_auto_ic_create(self):
        ts_ins = TSConstruct(self.reactant_ic, self.product_ic)
        ts_ins.auto_select_ic()
        ref_ic_rct = np.array([
            2.02762919, 2.02769736, 2.02761705, 1.77505755, -0.49059482,
            -0.49089531, -0.49066505, -0.9635587, 6.0521314, -0.07908926,
            -0.07896564, -0.0779316
        ])
        ref_ic_prd = np.array([
            2.03992597, 2.03991419, 2.03976417, 5.52444423, -0.33511068,
            -0.33461559, -0.33464313, -0.50318675, 2.6533652, -0.33183043,
            -0.33186034, -0.3319302
        ])
        assert np.allclose(ts_ins.reactant.ic_values, ref_ic_rct)
        assert np.allclose(ts_ins.product.ic_values, ref_ic_prd)

    def test_ts_construct(self):
        ts_ins = TSConstruct(self.reactant_ic, self.product_ic)
        ts_ins.auto_select_ic()
        ts_ins.create_ts_state(start_with="product")
        result = deepcopy(ts_ins.ts)
        # print result.ic_values
        ts_ins.create_ts_state(start_with="reactant")
        result_2 = deepcopy(ts_ins.ts)
        # print result_2.ic_values
        ref_tar_ic = ts_ins._reactant.ic_values * 0.5 + \
            ts_ins._product.ic_values * 0.5
        assert np.allclose(ref_tar_ic, result.target_ic)
        assert np.allclose(result.target_ic, result_2.target_ic)
        assert np.allclose(
            result.ic_values[:4], result.target_ic[:4], atol=1e-6)
        assert np.allclose(
            result_2.ic_values[:4], result_2.target_ic[:4], atol=1e-6)
        ts_ins.select_key_ic(1)
        assert ts_ins.key_ic_counter == 1
        assert np.allclose(ts_ins.ts.ic_values[:2][::-1],
                           result_2.ic_values[:2])
        assert np.allclose(ts_ins.ts._cc_to_ic_gradient[1],
                           result_2._cc_to_ic_gradient[0])

    def test_ts_union(self):
        self.reactant_ic.add_bond(0, 1)
        self.reactant_ic.add_bond(1, 2)
        self.reactant_ic.add_angle_cos(0, 1, 2)
        self.product_ic.add_bond(1, 0)
        self.product_ic.add_bond(0, 2)
        self.product_ic.add_angle_cos(1, 0, 2)
        ts_ins = TSConstruct(self.reactant_ic, self.product_ic)
        union_ic = ts_ins._get_union_of_ics()
        assert len(union_ic) == 5

    def test_ts_union_reactant(self):
        self.reactant_ic.add_bond(0, 1)
        self.reactant_ic.add_bond(1, 2)
        self.reactant_ic.add_angle_cos(0, 1, 2)
        ts_ins = TSConstruct(self.reactant_ic, self.product_ic)
        union_ic = ts_ins._get_union_of_ics(mode='reactant')
        assert len(union_ic) == 3
        ts_ins.auto_select_ic(auto_select=False, mode='reactant')
        assert len(ts_ins.product.ic) == 3

    def test_ts_union_product(self):
        self.product_ic.add_bond(1, 0)
        self.product_ic.add_bond(0, 2)
        self.product_ic.add_angle_cos(1, 0, 2)
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

    def test_ts_combine(self):
        self.reactant_ic.auto_select_ic()
        self.product_ic.auto_select_ic()
        new_ins = TSConstruct(self.reactant_ic, self.product_ic)
        new_ins.auto_generate_ts(start_with='product', reset_ic=False)
        ref_ic_rct = np.array([
            2.02762919, 2.02769736, 2.02761705, 1.77505755, -0.49059482,
            -0.49089531, -0.49066505, -0.9635587, 6.0521314, -0.07908926,
            -0.07896564, -0.0779316
        ])
        ref_ic_prd = np.array([
            2.03992597, 2.03991419, 2.03976417, 5.52444423, -0.33511068,
            -0.33461559, -0.33464313, -0.50318675, 2.6533652, -0.33183043,
            -0.33186034, -0.3319302
        ])
        assert np.allclose(new_ins.ts.ic_values[:4],
                           ((ref_ic_rct + ref_ic_prd) / 2)[:4])

    def test_choices_auto_select_ic(self):
        self.reactant_ic.add_bond(2, 4)
        new_ins = TSConstruct(self.reactant_ic, self.product_ic)
        new_ins.auto_generate_ts(auto_select=False)
        ref_ic = (
            self.reactant_ic.distance(2, 4) + self.product_ic.distance(2, 4)
        ) / 2
        assert np.allclose(new_ins.ts.ic_values, ref_ic)
        new_ins = TSConstruct(self.reactant_ic, self.product_ic)
        new_ins.auto_generate_ts(auto_select=True, reset_ic=False)
        assert len(new_ins.ts.ic) == 18
        assert np.allclose(new_ins.ts.ic_values[0], ref_ic, atol=1e-3)
        new_ins = TSConstruct(self.reactant_ic, self.product_ic)
        new_ins.auto_generate_ts(auto_select=True, reset_ic=True)
        ref_ic_rct = np.array([
            2.02762919, 2.02769736, 2.02761705, 1.77505755, -0.49059482,
            -0.49089531, -0.49066505, -0.9635587, 6.0521314, -0.07908926,
            -0.07896564, -0.0779316
        ])
        ref_ic_prd = np.array([
            2.03992597, 2.03991419, 2.03976417, 5.52444423, -0.33511068,
            -0.33461559, -0.33464313, -0.50318675, 2.6533652, -0.33183043,
            -0.33186034, -0.3319302
        ])
        assert np.allclose(new_ins.ts.ic_values[:4],
                           ((ref_ic_rct + ref_ic_prd) / 2)[:4])

    def test_from_file_and_to_file(self):
        rct_p = os.path.join(data_dir, "ch3_hf.xyz")
        prd_p = os.path.join(data_dir, "ch3f_h.xyz")
        ts = TSConstruct.from_file(rct_p, prd_p)
        ts_ins = TSConstruct(self.reactant_ic, self.product_ic)
        ts.auto_generate_ts()
        ts_ins.auto_generate_ts()
        filepath = os.path.join(data_dir, "ts_nose_test_cons.xyz")
        ts.ts_to_file(filepath)
        self.file_list.append(filepath)
        mol = IOData.from_file(filepath)
        assert np.allclose(mol.coordinates, ts.ts.coordinates)

    def test_from_file_to_path(self):
        rct_path = os.path.join(data_dir, "rct.xyz")
        prd_path = os.path.join(data_dir, "prd.xyz")
        ts_mol = TSConstruct.from_file(rct_path, prd_path)
        ts_mol.auto_generate_ts(task='path')
        assert isinstance(ts_mol.ts, Path_RI)

    @classmethod
    def tearDownClass(cls):
        for i in cls.file_list:
            os.remove(i)
