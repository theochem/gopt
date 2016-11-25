from copy import deepcopy

import os

import numpy as np

import horton as ht
from saddle.newopt.grape import Grape
from saddle.newopt.hessian_modifier import SaddleHessianModifier, Test_Saddle_Modifier
from saddle.newopt.saddle_point import SaddlePoint
from saddle.newopt.step_scaler import TRIM
from saddle.newopt.trust_radius import DefaultTrustRadius
from saddle.newopt.hessian_update import BFGS
from saddle.reduced_internal import ReducedInternal


class TestGrape(object):
    @classmethod
    def setup_class(self):
        fn_xyz = ht.context.get_fn("test/water.xyz")
        mol = ht.IOData.from_file(fn_xyz)  # create a water molecule
        self.ri = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)

    def test_minimun_water_from_other_ic(self):
        mol = deepcopy(self.ri)
        mol.add_bond(0, 1)
        mol.add_bond(1, 2)
        mol.add_angle_cos(0, 1, 2)
        mol.set_target_ic((2.4, 2.4, -0.5))
        mol.converge_to_target_ic()
        mol.set_key_ic_number(2)
        mol.energy_calculation()
        f_p = SaddlePoint(structure=mol)
        tr = DefaultTrustRadius(number_of_atoms=3)
        ss = TRIM()
        hm = Test_Saddle_Modifier()
        hu = BFGS()
        li_grape = Grape(
            hessian_update=hu,
            trust_radius=tr,
            step_scale=ss,
            hessian_modifier=hm)
        li_grape.add_point(f_p)
        #li_grape.modify_hessian(key_ic_number=0, negative_eigen=0)
        #li_grape.calculate_step(negative_eigen=0)
        # print li_grape.last.step
        # s_p = li_grape.calculate_new_point()
        #li_grape.update_to_new_point()
        #print li_grape.total
        #print li_grape.last.value
        #print li_grape.last._structure.ic_values
        #print np.linalg.norm(li_grape.last.gradient)
        li_grape.start_optimization(key_ic_number=2, iteration=20, init_hessian=False)


    def test_optimization_for_methanol(self):
        path = os.path.dirname(os.path.realpath(__file__))
        mol_path = path + '/../test/methanol.xyz'
        mol = ht.IOData.from_file(mol_path)
        methanol = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        methanol.auto_select_ic()
        methanol.energy_calculation()
        f_p = SaddlePoint(structure=methanol)
        tr = DefaultTrustRadius(number_of_atoms=5)
        ss = TRIM()
        hm = Test_Saddle_Modifier()
        hu = BFGS()
        li_grape = Grape(
            hessian_update=hu,
            trust_radius=tr,
            step_scale=ss,
            hessian_modifier=hm)
        li_grape.add_point(f_p)
        print methanol.ic
        print methanol.b_matrix.shape
        print (methanol.vspace.shape)
        assert False
