from unittest import TestCase

import numpy as np
from importlib_resources import path
from saddle.errors import NotSetError
from saddle.optimizer.react_point import ReactPoint
from saddle.reduced_internal import ReducedInternal
from saddle.ts_construct import TSConstruct
from saddle.utils import Utils


# pylint: disable=E1101, E1133
# Disable pylint on numpy.random functions
class TestReactPoint(TestCase):
    def setUp(self):
        with path('saddle.test.data', 'rct.xyz') as rct_path:
            with path('saddle.test.data', 'prd.xyz') as prd_path:
                mol = TSConstruct.from_file(rct_path, prd_path)
                mol.auto_generate_ts(dihed_special=True)
        self.ts = mol.ts
        self.dir_vec = mol.rct.ic_values - mol.prd.ic_values
        self.r_p1 = ReactPoint(self.ts, self.dir_vec)

    def test_dir_vec(self):
        b_matrix = self.ts.b_matrix
        proj_b = np.dot(self.ts.b_matrix, np.linalg.pinv(self.ts.b_matrix))
        # project twice
        dir_v = np.dot(proj_b, np.dot(proj_b, self.dir_vec))
        assert np.allclose(dir_v / np.linalg.norm(dir_v), self.r_p1.dir_vect)
        assert np.linalg.norm(self.r_p1.dir_vect) - 1 <= 1e-8

    def test_sub_vspace(self):
        proj_dv = np.outer(self.r_p1.dir_vect, self.r_p1.dir_vect)
        # project twice
        ref_sub_vspace = self.ts.vspace - np.dot(
            proj_dv, np.dot(proj_dv, self.ts.vspace))
        assert np.allclose(ref_sub_vspace, self.r_p1.vspace)

    def test_sub_v_gradient(self):
        # set random numpy seed
        np.random.seed(101)
        # set random gradient value
        x_gradient = np.random.rand(12)
        self.ts._energy_gradient = x_gradient
        ref_sub_v_gradient = np.dot(self.r_p1.vspace.T, self.ts.q_gradient)
        # r_p1 and ts have the same q_gradient
        assert np.allclose(self.r_p1.q_gradient, self.ts.q_gradient)
        # r_p1 has different v_gradient
        assert not np.allclose(self.r_p1.v_gradient, self.ts.v_gradient)
        assert np.allclose(self.r_p1.v_gradient, ref_sub_v_gradient)

    def test_sub_v_hessian(self):
        np.random.seed(111)
        # set random gradient
        x_gradient = np.random.rand(12)
        self.ts._energy_gradient = x_gradient
        # set random hessian
        x_hessian_h = np.random.rand(12, 12)
        x_hessian = np.dot(x_hessian_h, x_hessian_h.T)
        self.ts._energy_hessian = x_hessian

        ref_sub_v_hessian = np.dot(self.r_p1.vspace.T,
                                   np.dot(self.ts.q_hessian, self.r_p1.vspace))
        assert np.allclose(self.r_p1.q_gradient, self.ts.q_gradient)
        assert np.allclose(self.r_p1._instance.q_hessian, self.ts.q_hessian)
        assert np.allclose(self.r_p1.v_hessian, ref_sub_v_hessian)

    def test_x_and_q_gradient(self):
        np.random.seed(212)
        # set random gradient
        x_gradient = np.random.rand(12)
        self.ts._energy_gradient = x_gradient
        # set random hessian
        ref_sub_q_g = np.dot(self.r_p1.vspace, self.r_p1.v_gradient)
        assert np.allclose(ref_sub_q_g, self.r_p1.sub_q_gradient)
        ref_sub_x_g = np.dot(self.r_p1.b_matrix.T, self.r_p1.sub_q_gradient)
        assert np.allclose(ref_sub_x_g, self.r_p1.sub_x_gradient)
