import os

import numpy as np

import horton as ht
from saddle.reduced_internal import ReducedInternal
from saddle.newopt.grape import Grape
from saddle.newopt.hessian_modifier import SaddleHessianModifier
from saddle.newopt.hessian_update import BFGS, PSB, SR1
from saddle.newopt.saddle_point import SaddlePoint
from saddle.newopt.step_scaler import TRIM
from saddle.newopt.trust_radius import DefaultTrustRadius


class TestGrape(object):
    @classmethod
    def setup_class(self):
        fn_xyz = ht.context.get_fn("test/water.xyz")
        mol = ht.IOData.from_file(fn_xyz)  # create a water molecule
        self.ri = ReducedInternal(mol.coordinates, mol.numbers, 0, 2)
        path = os.path.dirname(os.path.realpath(__file__))
        fchk_path = path + "/../../test/s_p_0.fchk"
        self.ri.add_bond(0, 1)
        self.ri.add_bond(1, 2)
        self.ri.add_angle_cos(0, 1, 2)
        self.ri.set_target_ic([2.0, 2.0, -0.5])
        self.ri.converge_to_target_ic()
        self.ri.energy_from_fchk(fchk_path)

    def test_obtained_energy(self):
        assert self.ri.energy == -75.97088850192669
        assert np.allclose(self.ri.ic_values, [2., 2., -0.49999995])
        assert np.allclose(self.ri.internal_gradient, np.dot(
            np.linalg.pinv(self.ri.b_matrix.T), self.ri.energy_gradient))
        assert np.allclose(self.ri.vspace_gradient,
                           np.dot(self.ri.vspace.T, self.ri.internal_gradient))
        assert np.allclose(np.linalg.norm(self.ri.vspace), np.sqrt(3))

    def test_new_point(self):
        s_p_0 = SaddlePoint(self.ri)
        assert isinstance(s_p_0, SaddlePoint)
        hm = SaddleHessianModifier()
        hu = SR1()
        ss = TRIM()
        tr = DefaultTrustRadius(3)
        optser = Grape(tr, hu, ss, hm)
        optser.add_point(s_p_0)
        optser.modify_hessian(key_ic_number=0, negative_eigen=0)
        optser.calculate_step(negative_eigen=0)
        print
        s_p_1 = optser.last.update_point()
        path = os.path.dirname(os.path.realpath(__file__))
        fchk_path_1 = path + "/../../test/s_p_1.fchk"
        s_p_1._structure.energy_from_fchk(fchk_path_1)
        s_p_1._structure.align_vspace(s_p_0._structure)
        assert not np.allclose(s_p_0.hessian, s_p_1.hessian)
        assert np.allclose(s_p_1.structure.ic_values,
                           [1.6708903, 1.6708903, -0.40384809])
        assert np.allclose(s_p_1.structure.energy, -75, 98276816991029)
        assert np.allclose(s_p_1.structure.vspace, s_p_0.structure.vspace)
        optser.add_point(s_p_1)

        s_0 = optser._points[0]
        s_1 = optser._points[1]

        y = (s_1.gradient - s_0.gradient) - np.dot(
            np.dot(s_1.structure.vspace.T,
                   np.linalg.pinv(s_1.structure.b_matrix.T)), np.dot(
                       (s_1.structure.b_matrix - s_0.structure.b_matrix).T,
                       s_1.structure.internal_gradient))  #secant condition
        secant_cond = optser._h_u.secant_condition(s_0.structure,
                                                   s_1.structure)
        assert np.allclose(y, secant_cond)
        change = y - np.dot(s_0.hessian, s_0.step)
        h_ref = s_0.hessian + np.outer(change, change) / np.dot(change,
                                                                s_0.step)
        h_new = optser._h_u.update_hessian(s_0, s_1)

        assert np.allclose(h_new, h_ref)

        # change hessian update method to PSB
        optser._h_u = PSB()
        h_ref = s_0.hessian + (np.outer(change, s_0.step) + np.outer(
            s_0.step, change)) / np.dot(s_0.step, s_0.step) - np.dot(
                s_0.step, change) / np.dot(s_0.step, s_0.step)**2 * np.outer(
                    s_0.step, s_0.step)
        h_new = optser._h_u.update_hessian(s_0, s_1)
        assert np.allclose(h_ref, h_new)

        # change hessian update method to BFGS
        optser._h_u = BFGS()
        temp_g = np.dot(s_0.hessian, s_0.step)
        h_ref = s_0.hessian + np.outer(y, y) / np.dot(y, s_0.step) - np.outer(
            temp_g, temp_g) / np.dot(s_0.step, temp_g)
        h_new = optser._h_u.update_hessian(s_0, s_1)
        assert np.allclose(h_ref, h_new)
        assert not np.allclose(s_1.hessian, h_ref)
        optser.update_hessian()
        assert np.allclose(s_1.hessian, h_ref)
