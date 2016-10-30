import numpy as np

from copy import deepcopy

import horton as ht
from saddle.newopt.grape import Grape
from saddle.newopt.hessian_modifier import SaddleHessianModifier
from saddle.newopt.saddle_point import SaddlePoint
from saddle.newopt.step_scaler import TRIM
from saddle.newopt.trust_radius import DefaultTrustRadius
from saddle.reduced_internal import ReducedInternal


class TestGrape(object):
    @classmethod
    def setup_class(self):
        fn_xyz = ht.context.get_fn("test/water.xyz")
        mol = ht.IOData.from_file(fn_xyz)  # create a water molecule
        self.ri = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)

    def test_minimum_optimization(self):
        mol = deepcopy(self.ri)
        mol.add_bond(0, 1)
        mol.add_bond(1, 2)
        mol.add_angle_cos(0, 1, 2)
        mol.energy_calculation()
        f_p = SaddlePoint(structure=mol)
        print (f_p.value)
        tr = DefaultTrustRadius(number_of_atoms=3)
        ss = TRIM()
        hm = SaddleHessianModifier()
        li_grape = Grape(
            hessian_update=None,
            trust_radius=tr,
            step_scale=ss,
            hessian_modifier=hm)
        li_grape.add_point(f_p)
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
	print li_grape.last.step
        s_p = li_grape.calculate_new_point()
        s_p._structure.energy_calculation()
        print (s_p.value)
        li_grape.add_point(s_p)
        li_grape._t_r.update(li_grape.last,li_grape._points[-2],'energy')
        print np.linalg.norm(li_grape._points[-2].step)
        print (li_grape._points[-2].trust_radius_stride)
        print li_grape.last.trust_radius_stride
        assert False
