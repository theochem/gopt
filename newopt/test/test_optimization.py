from copy import deepcopy

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
        mol.set_target_ic((2.0, 2.0, -0.2))
        mol.converge_to_target_ic()
        mol.set_key_ic_number(0)
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
        li_grape.start_optimization(iteration=10)


'''
    def test_minimun_water_from_other_ic(self):
        mol = deepcopy(self.ri)
        mol.add_bond(0, 1)
        mol.add_bond(1, 2)
        mol.add_angle_cos(0, 1, 2)
        mol.set_target_ic((2.0, 2.0, -0.2))
        mol.converge_to_target_ic()
        print mol.ic_values
        assert np.allclose(mol.ic_values, (2.0, 2.0, -0.2), atol=1e-4)
        mol.energy_calculation()
        f_p = SaddlePoint(structure=mol)
        tr = DefaultTrustRadius(number_of_atoms=3)
        ss = TRIM()
        hm = SaddleHessianModifier()
        li_grape = Grape(
            hessian_update=None,
            trust_radius=tr,
            step_scale=ss,
            hessian_modifier=hm)
        li_grape.add_point(f_p)
        li_grape.modify_hessian(key_ic_number=0, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        # print li_grape.last.step
        # s_p = li_grape.calculate_new_point()
        li_grape.update_to_new_point()
        print li_grape.total
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        print li_grape.last._structure.energy_gradient
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=0, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        print li_grape.last._structure.energy_gradient
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=0, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        print li_grape.last._structure.energy_gradient
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=0, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        print li_grape.last._structure.energy_gradient
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=0, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        print li_grape.last._structure.energy_gradient
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=0, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        print li_grape.last._structure.energy_gradient
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=0, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        print li_grape.last._structure.energy_gradient
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=0, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        print li_grape.last._structure.energy_gradient
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=0, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        print li_grape.last._structure.energy_gradient
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=0, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        print li_grape.last._structure.energy_gradient
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=0, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        print li_grape.last._structure.energy_gradient
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=0, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        print li_grape.last._structure.energy_gradient
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=0, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        print li_grape.last._structure.energy_gradient
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=0, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        print li_grape.last._structure.energy_gradient
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        assert False
'''
'''
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
        print li_grape.last._structure.ic_values
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
	print np.linalg.norm(li_grape.last.gradient)
'''
        

'''
    def test_minimum_optimization(self):
        mol = deepcopy(self.ri)
        mol.add_bond(0, 1)
        mol.add_bond(1, 2)
        mol.add_angle_cos(0, 1, 2)
        mol.energy_calculation()
        f_p = SaddlePoint(structure=mol)
        print(f_p.value)
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
        # s_p = li_grape.calculate_new_point()
        li_grape.update_to_new_point()
        # s_p._structure.energy_calculation()
        # print(s_p.value)
        # li_grape.add_point(s_p)
        # li_grape._t_r.update(li_grape.last, li_grape._points[-2], 'energy')
        # print np.linalg.norm(li_grape._points[-2].step)
        # print(li_grape._points[-2].trust_radius_stride)
        # print li_grape.last.trust_radius_stride
        print li_grape.total
        print li_grape.last.value
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
	print np.linalg.norm(li_grape.last.gradient)
        li_grape.update_trust_radius(criterion="energy")
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=0)
        li_grape.calculate_step(negative_eigen=0)
        print li_grape.last.step
        li_grape.update_to_new_point()
        print li_grape.last.value
	print np.linalg.norm(li_grape.last.gradient)
'''
