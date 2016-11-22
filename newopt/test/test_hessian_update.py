import os

import numpy as np

import horton as ht

from saddle.newopt.hessian_update import *
from saddle.newopt.grape import Grape
from saddle.newopt.saddle_point import SaddlePoint
from saddle.newopt.hessian_modifier import SaddleHessianModifier
from saddle.newopt.hessian_update import BFGS
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
        assert np.allclose(self.ri._internal_gradient, np.dot(
            np.linalg.pinv(self.ri._cc_to_ic_gradient.T),
            self.ri.energy_gradient))
        assert np.allclose(self.ri.vspace_gradient, np.dot(
            self.ri.vspace.T, self.ri._internal_gradient))
        assert np.allclose(np.linalg.norm(self.ri.vspace), np.sqrt(3))

    def test_new_point(self):
        s_p_0 = SaddlePoint(self.ri)
        assert isinstance(s_p_0, SaddlePoint)
        hm = SaddleHessianModifier()
        hu = BFGS()
        ss = TRIM()
        tr = DefaultTrustRadius(3)
        optser = Grape(tr, hu, ss, hm)
        optser.add_point(s_p_0)
        optser.modify_hessian(key_ic_number=0, negative_eigen=0)
        optser.calculate_step(negative_eigen=0)
        s_p_1 = optser.last.update_point()
        path = os.path.dirname(os.path.realpath(__file__))
        fchk_path_1 = path + "/../../test/s_p_1.fchk"
        s_p_1._structure.energy_from_fchk(fchk_path_1)
        s_p_1._structure.align_vspace(s_p_0._structure)
        assert not np.allclose(s_p_0.hessian, s_p_1.hessian)
        assert np.allclose(s_p_1.structure.ic_values,
                           [1.6708903, 1.6708903, -0.40384809])
        assert np.allclose(s_p_1._structure.energy, -75, 98276816991029)
        assert np.allclose(s_p_1._structure.vspace, s_p_0._structure.vspace)
        optser.add_point(s_p_1)
