import unittest

import numpy as np
from saddle.errors import PositiveProductError
from saddle.math_lib import diagonalize, pse_inv, ridders_solver


class TestSolver(unittest.TestCase):
    def test_ridder_quadratic(self):
        func = lambda x: x**2 - 4
        answer = ridders_solver(func, -1, 10)
        assert abs(answer - 2) < 1e-6

        try:
            ridders_solver(func, -3, 3)
        except PositiveProductError:
            assert True
        else:
            assert False

    def test_ridder_quatic(self):
        func = lambda x: x**3 - x**2 - 2 * x
        answer = ridders_solver(func, -2, 0)
        assert answer == 0

        answer = ridders_solver(func, -2, -0.5)
        assert abs(answer - (-1)) < 1e-6

        answer = ridders_solver(func, -0.5, 1)
        assert abs(answer - 0) < 1e-6

        answer = ridders_solver(func, 1.5, 3)
        assert abs(answer - 2) < 1e-6

    def test_diagonalize(self):
        np.random.seed(111)
        mtx = np.random.rand(4, 2)
        assert mtx.shape == (4, 2)
        ei_value, ei_vector = diagonalize(mtx)
        prd = np.dot(mtx, mtx.T)
        assert prd.shape == (4, 4)
        v, w = np.linalg.eigh(prd)
        assert np.allclose(ei_value, v)
        assert np.allclose(ei_vector, w)

    def test_pse_inv(self):
        np.random.seed(500)
        mtr_1 = np.random.rand(5, 5)
        mtr_1_r = pse_inv(mtr_1)
        assert np.allclose(np.dot(mtr_1, mtr_1_r), np.eye(5))

        mtr_2 = np.array([[3, 0], [0, 0]])
        mtr_2_r = pse_inv(mtr_2)
        ref_mtr = np.array([[1/3, 0], [0, 0]])
        assert np.allclose(mtr_2_r, ref_mtr)

        mtr_3 = np.array([[3, 3e-9], [1e-11, 2e-10]])
        mtr_3_r = pse_inv(mtr_3)
        assert np.allclose(mtr_3_r, ref_mtr)

    def test_pse_inv_with_np(self):
        np.random.seed(100)
        for _ in range(5):
            shape = np.random.randint(1, 10, 2)
            target_mt = np.random.rand(*shape)
            np_ref = np.linalg.pinv(target_mt)
            pse_inv_res = pse_inv(target_mt)
            assert np.allclose(pse_inv_res, np_ref)

    def test_pse_inv_close(self):
        np.random.seed(200)
        for _ in range(5):
            shape = np.random.randint(1, 20, 2)
            rand_mt = np.random.rand(*shape)
            inv_mt = pse_inv(rand_mt)
            diff = np.dot(np.dot(rand_mt, inv_mt), rand_mt) - rand_mt
            assert np.allclose(np.linalg.norm(diff), 0)
