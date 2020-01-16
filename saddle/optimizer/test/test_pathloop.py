from unittest import TestCase

from copy import deepcopy
import numpy as np
from importlib_resources import path
from saddle.optimizer.pathloop import PathLoop
from saddle.optimizer.react_point import ReactPoint
from saddle.optimizer.hessian_modify import modify_hessian_with_pos_defi
from saddle.ts_construct import TSConstruct
from saddle.optimizer.trust_radius import TrustRegion


class TestPathLoop(TestCase):
    def setUp(self):
        with path("saddle.optimizer.test.data", "HNCS.xyz") as rct_f:
            with path("saddle.optimizer.test.data", "HSCN.xyz") as prd_f:
                ts_cons = TSConstruct.from_file(rct_f, prd_f)
        ts_cons.auto_generate_ts(dihed_special=True)
        self.ts = ts_cons.ts
        self.dir_vec = ts_cons.prd.ic_values - ts_cons.rct.ic_values
        with path("saddle.optimizer.test.data", "pathloop_hscn.fchk") as fchk_f:
            self.ts.energy_from_fchk(fchk_f)
        self.opt_ob = PathLoop(
            self.ts, self.dir_vec, quasi_nt="bfgs", trust_rad="trim", upd_size="energy"
        )

    def test_init(self):
        assert isinstance(self.opt_ob, PathLoop)
        assert isinstance(self.opt_ob[0], ReactPoint)

    def test_first_step(self):
        dpcp_p = deepcopy(self.opt_ob[0])
        # build self v
        proj_b = np.dot(self.ts.b_matrix, np.linalg.pinv(self.ts.b_matrix))
        dir_vec = np.dot(proj_b, self.dir_vec)
        unit_dv = dir_vec / np.linalg.norm(dir_vec)
        proj_dv = np.outer(unit_dv, unit_dv)
        sub_v = self.ts.vspace - np.dot(proj_dv, self.ts.vspace)
        # test vspace
        assert np.allclose(sub_v, self.opt_ob[0].vspace)
        v_hessian = np.dot(np.dot(sub_v.T, self.ts.q_hessian), sub_v)
        # test v hessian
        assert np.allclose(v_hessian, self.opt_ob[0].v_hessian)
        new_sub_v = modify_hessian_with_pos_defi(v_hessian, 0, 0)

        # modify_hessian and test
        self.opt_ob.modify_hessian()
        assert np.allclose(self.opt_ob[0].step_hessian, new_sub_v)

        # calculate step
        ref_step = TrustRegion.trim(
            new_sub_v, self.opt_ob[0].v_gradient, self.opt_ob[0].stepsize
        )
        self.opt_ob.calculate_trust_step()
        assert np.allclose(self.opt_ob[0].step, ref_step)

        # calculate next point
        new_p = self.opt_ob.next_step_structure()
        assert isinstance(new_p, ReactPoint)
        dpcp_p.update_coordinates_with_delta_v(ref_step)
        assert np.allclose(new_p._instance.coordinates, dpcp_p._instance.coordinates)
