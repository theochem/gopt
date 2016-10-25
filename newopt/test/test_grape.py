import numpy as np

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
        self.ri.add_bond(0, 1)
        self.ri.add_bond(1, 2)
        self.ri.add_angle_cos(0, 1, 2)
        self.ri.set_key_ic_number(1)
        self.ri.set_new_coordinates(
            np.array([[0.00000000e+00, 1.48124293e+00, -8.37919685e-01], [
                0.00000000e+00, 3.42113883e-49, 2.09479921e-01
            ], [-1.81399942e-16, -1.48124293e+00, -8.37919685e-01]]))
        self.ri._energy = -76.3856892935073
        self.ri._energy_gradient = np.array(
            [1.79016197e-17, -1.14393578e-02, 8.56031577e-03, 7.42919139e-16,
             2.13370988e-16, -1.71206315e-02, -7.60820759e-16, 1.14393578e-02,
             8.56031577e-03])
        self.ri._energy_hessian = np.array(
            [[-7.99278901e-03, 1.13049265e-13, -1.69204779e-13, 8.20209517e-03,
              -1.23901570e-12, -1.48730466e-12, -2.09306165e-04,
              1.13811665e-12, 3.10484455e-12],
             [1.13049265e-13, 4.02310139e-01, -2.30853436e-01, -4.72243644e-13,
              -3.71668821e-01, 1.93062118e-01, 1.13811406e-12, -3.06413180e-02,
              3.77913184e-02],
             [-1.69204779e-13, -2.30853436e-01, 1.87408974e-01, 2.43251039e-12,
              2.68644754e-01, -1.92593033e-01, -3.10484110e-12,
              -3.77913184e-02, 5.18405956e-03],
             [8.20209517e-03, -4.72243644e-13, 2.43251039e-12, -1.64041903e-02,
              1.35624049e-12, -2.62930325e-12, 8.20209517e-03, -2.37809080e-13,
              -2.10845134e-12],
             [-1.23901570e-12, -3.71668821e-01, 2.68644754e-01, 1.35624049e-12,
              7.43337643e-01, 4.71279266e-11, -1.90889518e-12, -3.71668821e-01,
              -2.68644754e-01], [
                  -1.48730466e-12, 1.93062118e-01, -1.92593033e-01,
                  -2.62930325e-12, 4.71279266e-11, 3.85186067e-01,
                  3.79292664e-12, -1.93062118e-01, -1.92593033e-01
              ], [2.09306165e-04, 1.13811406e-12, -3.10484110e-12,
                  8.20209517e-03, -1.90889518e-12, 3.79292664e-12,
                  -7.99278901e-03, 1.12443136e-13, 1.68823548e-13], [
                      1.13811665e-12, -3.06413180e-02, -3.77913184e-02,
                      -2.37809080e-13, -3.71668821e-01, -1.93062118e-01,
                      1.12443136e-13, 4.02310139e-01, 2.30853436e-01
                  ], [3.10484455e-12, 3.77913184e-02, 5.18405956e-03,
                      -2.10845134e-12, -2.68644754e-01, -1.92593033e-01,
                      1.68823548e-13, 2.30853436e-01, 1.87408974e-01]])

        self.ri._internal_gradient = np.array(
            [-0.0142825, -0.0142825, -0.00074069])
        self.ri._internal_hessian = np.array(
            [[0.54831182, -0.01345925, -0.04730117],
             [-0.01345925, 0.54831182, -0.04730117],
             [-0.04730117, -0.04730117, 0.18274632]])
        self.ri._vspace_gradient = np.dot(self.ri.vspace.T,
                                          self.ri._internal_gradient)
        self.ri._vspace_hessian = np.dot(
            np.dot(self.ri.vspace.T, self.ri._internal_hessian),
            self.ri.vspace)

    def test_vspace_value(self):
        test_x_step = np.array([0.1, 0., 0., 0., 0.2, 0., 0., 0.1, 0.1])
        test_ic_step = np.dot(self.ri._cc_to_ic_gradient, test_x_step)
        test_v_step = np.dot(self.ri.vspace.T, test_ic_step)
        g_x = np.dot(self.ri.energy_gradient, test_x_step)
        g_ic = np.dot(self.ri._internal_gradient, test_ic_step)
        g_v = np.dot(self.ri._vspace_gradient, test_v_step)
        assert np.allclose(g_x, g_ic)
        assert np.allclose(g_ic, g_v)

    def test_optimizer_initializataion(self):
        f_p = SaddlePoint(structure=self.ri)
        tr = DefaultTrustRadius(number_of_atoms=3)
        ss = TRIM()
        hm = SaddleHessianModifier()
        li_grape = Grape(
            hessian_update=None,
            trust_radius=tr,
            step_scale=ss,
            hessian_modifier=hm)
        li_grape.add_point(f_p)
        assert np.allclose(f_p.hessian, self.ri._vspace_hessian)
        # print(f_p.hessian)
        li_grape.modify_hessian(key_ic_number=1, negative_eigen=1)
        # print li_grape._points[0].hessian
        # print(f_p.hessian)
        p_w, _ = np.linalg.eigh(f_p.hessian)
        w, _ = np.linalg.eigh(li_grape._points[0].hessian)
        assert np.allclose(w, np.array([-0.005, 0.17046595, 0.54713294]))
        assert np.allclose(p_w, np.array([0.17046595, 0.54713294, 0.56177107]))
        assert li_grape.total == 1
        assert np.allclose(li_grape.last.trust_radius_stride,
                           1.7320508075688772 * 0.35)
        li_grape.calculate_step(negative_eigen=1)
        assert (np.linalg.norm(li_grape.last.step) <=
                li_grape.last.trust_radius_stride)
        # assert False
