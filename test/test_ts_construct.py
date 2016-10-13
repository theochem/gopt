import horton as ht
import numpy as np
import os
from saddle.ts_construct import TSConstruct
from saddle.internal import Internal


class Test_TS_Construct(object):
    @classmethod
    def setup_class(self):
        path = os.path.dirname(os.path.realpath(__file__))
        self.rct = ht.IOData.from_file(path + "/ch3_hf.xyz")
        self.prd = ht.IOData.from_file(path + "/ch3f_h.xyz")

    def test_create_instance(self):
        reactant_ic = Internal(self.rct.coordinates, self.rct.numbers, 0, 2)
        product_ic = Internal(self.prd.coordinates, self.prd.numbers, 0, 2)
        ts_ins = TSConstruct(reactant_ic, product_ic)
        assert isinstance(ts_ins.reactant, Internal)
        assert isinstance(ts_ins.product, Internal)
        ts_ins._auto_select_bond()
        ts_ins._auto_select_angle()
        ts_ins._auto_select_dihed_normal()
        assert len(ts_ins.reactant.ic) == len(ts_ins.product.ic)
        print ts_ins.product.ic
        # print ts_ins.product.ic_values
        ref_value = [
            2.0399259678469988, 2.0399141882105529, 2.03976416977833,
            2.6533652025753693, 5.5244442325320069, -0.33511068208525152,
            -0.33461558852288653, -0.33183042590229783, -0.33464313367330611,
            -0.33186033997149444, -0.33193020192434486, -0.99936959502185685,
            -0.072069934158273591, -0.8278604118789985, 0.89987918554562152
        ]
        assert np.allclose(ts_ins.product.ic_values, ref_value)
        ts_ins._auto_select_dihed_improper()
        print ts_ins.product.ic
        ref_value = [2.0399259678469988, 2.0399141882105529, 2.03976416977833,
                     2.6533652025753693, 5.5244442325320069,
                     -0.33511068208525152, -0.33461558852288653,
                     -0.33183042590229783, -0.33464313367330611,
                     -0.33186033997149444, -0.33193020192434486,
                     -0.99936959502185685, -0.072069934158273591,
                     -0.8278604118789985, 0.89987918554562152, -0.503186754184]
        assert np.allclose(ts_ins.product.ic_values, ref_value)

    def test_auto_ic_create(self):
        reactant_ic = Internal(self.rct.coordinates, self.rct.numbers, 0, 2)
        product_ic = Internal(self.prd.coordinates, self.prd.numbers, 0, 2)
        ts_ins = TSConstruct(reactant_ic, product_ic)
        ts_ins.auto_select_ic()
        ref_ic = [2.0399259678469988, 2.0399141882105529, 2.03976416977833,
                  2.6533652025753693, 5.5244442325320069, -0.33511068208525152,
                  -0.33461558852288653, -0.33183042590229783,
                  -0.33464313367330611, -0.33186033997149444,
                  -0.33193020192434486, -0.99936959502185685,
                  -0.072069934158273591, -0.8278604118789985,
                  0.89987918554562152, -0.503186754184]
        assert np.allclose(ts_ins.product.ic_values, ref_ic)

    def test_ts_construct(self):
        from copy import deepcopy
        reactant_ic = Internal(self.rct.coordinates, self.rct.numbers, 0, 2)
        product_ic = Internal(self.prd.coordinates, self.prd.numbers, 0, 2)
        ts_ins = TSConstruct(reactant_ic, product_ic)
        ts_ins.auto_select_ic()
        ts_ins.create_ts_state(start_with="product")
        result = deepcopy(ts_ins.ts)
        # print result.ic_values
        ts_ins.create_ts_state(start_with="reactant")
        result_2 = deepcopy(ts_ins.ts)
        # print result_2.ic_values
        ref_tar_ic = reactant_ic.ic_values * 0.5 + product_ic.ic_values * 0.5
        assert np.allclose(ref_tar_ic, result.target_ic)
        assert np.allclose(result.target_ic, result_2.target_ic)
        assert np.allclose(
            result.ic_values[:5], result.target_ic[:5], atol=1e-7)
        assert np.allclose(
            result_2.ic_values[:5], result_2.target_ic[:5], atol=1e-7)
        ts_ins.select_key_ic(1)
        assert ts_ins.key_ic_counter == 1
        assert np.allclose(ts_ins.ts.ic_values[:2][::-1],
                           result_2.ic_values[:2])
        assert np.allclose(ts_ins.ts._cc_to_ic_gradient[1],
                           result_2._cc_to_ic_gradient[0])
