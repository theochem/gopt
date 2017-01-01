import os
from copy import deepcopy

import numpy as np

import horton as ht

from saddle.internal import Internal
from saddle.ts_construct import TSConstruct
from saddle.newopt.grape import Grape
from saddle.newopt.hessian_modifier import (SaddleHessianModifier,
                                            Test_Saddle_Modifier)
from saddle.newopt.hessian_update import BFGS
from saddle.newopt.saddle_point import SaddlePoint
from saddle.newopt.step_scaler import TRIM
from saddle.newopt.trust_radius import DefaultTrustRadius


class TestTS(object):
    def test_br_h_cl(self):
        path = os.path.dirname(os.path.realpath(__file__))
        reactant = path + '/../../test/Br_HCl.xyz'
        product = path + '/../../test/Cl_HBr.xyz'
        rct_mol = ht.IOData.from_file(reactant)
        prd_mol = ht.IOData.from_file(product)
        rct_ic = Internal(
            rct_mol.coordinates, rct_mol.numbers, charge=0, spin=2)
        prd_ic = Internal(
            prd_mol.coordinates, prd_mol.numbers, charge=0, spin=2)
        ts_cons = TSConstruct(rct_ic, prd_ic)
        ts_cons.auto_generate_ts()
        ts_structure = deepcopy(ts_cons.ts)
        assert np.allclose(ts_structure.ic_values,
                           [4.00695734, 3.90894008, -1.])
        print(type(ts_structure))
        ts_structure.set_key_ic_number(2)
        assert ts_structure.key_ic_number == 2
        # assert False
        ts_structure.energy_calculation()
        first_p = SaddlePoint(structure=ts_structure)
        tr = DefaultTrustRadius(number_of_atoms=3)
        ss = TRIM()
        hm = Test_Saddle_Modifier()
        hu = BFGS()
        li_grape = Grape(
            hessian_update=hu,
            trust_radius=tr,
            step_scale=ss,
            hessian_modifier=hm)
        li_grape.add_point(first_p)
        li_grape.start_optimization(
            iteration=20, key_ic_number=2, negative_eigen=1)
