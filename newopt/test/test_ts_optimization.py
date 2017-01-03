import os
from copy import deepcopy

import numpy as np

import horton as ht
from saddle.internal import Internal
from saddle.newopt.grape import Grape
from saddle.newopt.hessian_modifier import (SaddleHessianModifier,
                                            Test_Saddle_Modifier)
from saddle.newopt.hessian_update import BFGS
from saddle.newopt.saddle_point import SaddlePoint
from saddle.newopt.step_scaler import TRIM
from saddle.newopt.trust_radius import DefaultTrustRadius
from saddle.ts_construct import TSConstruct


class TestTS(object):
    def test_br_ch3_cl(self):
        path = os.path.dirname(os.path.realpath(__file__))
        fn_xyz = path + '/../../test/cl_ch3_br.xyz'
        mol = ht.IOData.from_file(fn_xyz)
        red_mol = ReducedInternal(
            mol.coordinates, mol.numbers, spin=1, charge=-1)
        red_mol.auto_select_ic()
        red_mol.delete_ic(5, 6)
        red_mol.auto_select_ic(keep_bond=True)
        assert len(red_mol) == 19
        red_mol.select_key_ic(0, 2)
        # product = path + '/../../test/Cl_HBr.xyz'
        # rct_mol = ht.IOData.from_file(reactant)
        # prd_mol = ht.IOData.from_file(product)
        # rct_ic = Internal(
        #     rct_mol.coordinates, rct_mol.numbers, charge=0, spin=2)
        # prd_ic = Internal(
        #     prd_mol.coordinates, prd_mol.numbers, charge=0, spin=2)
        # ts_cons = TSConstruct(rct_ic, prd_ic)
        # ts_cons.auto_generate_ts()
        # ts_structure = deepcopy(ts_cons.ts)
        # assert np.allclose(ts_structure.ic_values,
        #                    [4.00695734, 3.90894008, -1.])
        # print(type(ts_structure))
        # ts_structure.set_key_ic_number(2)
        # assert ts_structure.key_ic_number == 2
        # # assert False
        red_mol.energy_calculation()
        first_p = SaddlePoint(structure=red_mol)
        tr = DefaultTrustRadius(number_of_atoms=6)
        ss = TRIM()
        hm = Test_Saddle_Modifier()
        hu = BFGS()
        li_grape = Grape(
            hessian_update=hu,
            trust_radius=tr,
            step_scale=ss,
            hessian_modifier=hm)
        li_grape.add_point(first_p)
        li_grape.modify_hessian(key_ic_number=2, negative_eigen=1)
        li_grape.calculate_step(negative_eigen=1)
        print li_grape.last.step
        s_p = li_grape.calculate_new_point()
        li_grape.update_to_new_point()
