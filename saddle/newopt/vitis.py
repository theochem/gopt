from __future__ import absolute_import, division, print_function

import numpy as np

from saddle.newopt.grape import Grape
from saddle.newopt.saddle_point import SaddlePoint


class Vitis(Grape):
    @staticmethod
    def _points_accept_criterion(new_p, old_p):
        n_hyper_g = Vitis._projection(new_p)

        o_hyper_g = Vitis._projection(old_p)

        norm = np.linalg.norm
        return (norm(n_hyper_g) < norm(o_hyper_g))

    def converge_test(self, g_cutoff=1e-4, e_cutoff=1e-6, *args, **kwargs):
        final_p = self.last
        pre_p = self._points[-2]
        f_hyper_g = Vitis._projection(
            final_p)  # this may cause a bug when inherit
        if np.max(np.abs(f_hyper_g)) < g_cutoff:
            return True
        if np.abs(final_p.value - pre_p.value) < e_cutoff:
            return True
        # elif np.max(np.abs(pre_p.step)) < 3e-4:
        #    return True
        return False

    @staticmethod
    def _projection(point):
        assert isinstance(point, SaddlePoint)
        p_s = point.structure
        p_g = p_s.energy_gradient
        p_b = p_s.b_matrix
        p_v = p_s.vspace
        transform = np.dot(
            np.dot(p_b.T, p_v), np.dot(p_v.T, np.linalg.pinv(p_b.T)))
        return np.dot(transform, p_g)
