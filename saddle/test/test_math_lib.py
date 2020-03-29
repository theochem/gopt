import unittest

import numpy as np
from numpy.testing import assert_allclose
from saddle.errors import PositiveProductError
from saddle.math_lib import (
    diagonalize,
    pse_inv,
    ridders_solver,
    maximum_overlap,
    procrustes,
)


# pylint: disable=E1101, E1133
# Disable pylint on numpy.random functions
class TestSolver(unittest.TestCase):
    def test_ridder_quadratic(self):
        def func(x):
            return x ** 2 - 4

        answer = ridders_solver(func, -1, 10)
        assert abs(answer - 2) < 1e-6

        try:
            ridders_solver(func, -3, 3)
        except PositiveProductError:
            assert True
        else:
            assert False

    def test_ridder_quatic(self):
        def func(x):
            return x ** 3 - x ** 2 - 2 * x

        answer = ridders_solver(func, -2, 0)
        assert answer == 0

        answer = ridders_solver(func, -2, -0.5)
        assert abs(answer - (-1)) < 1e-6

        answer = ridders_solver(func, -0.5, 1)
        assert abs(answer - 0) < 1e-6

        answer = ridders_solver(func, 1.5, 3)
        assert abs(answer - 2) < 1e-6

    def test_diagonalize(self):
        # np.random.seed(111)
        mtx = np.random.rand(4, 2)
        assert mtx.shape == (4, 2)
        ei_value, ei_vector = diagonalize(mtx)
        prd = np.dot(mtx, mtx.T)
        assert prd.shape == (4, 4)
        v, w = np.linalg.eigh(prd)
        assert np.allclose(ei_value, v)
        assert np.allclose(ei_vector, w)

    def test_pse_inv(self):
        # np.random.seed(500)
        mtr_1 = np.random.rand(5, 5)
        mtr_1_r = pse_inv(mtr_1)
        assert np.allclose(np.dot(mtr_1, mtr_1_r), np.eye(5))

        mtr_2 = np.array([[3, 0], [0, 0]])
        mtr_2_r = pse_inv(mtr_2)
        ref_mtr = np.array([[1 / 3, 0], [0, 0]])
        assert np.allclose(mtr_2_r, ref_mtr)

        mtr_3 = np.array([[3, 3e-9], [1e-11, 2e-10]])
        mtr_3_r = pse_inv(mtr_3)
        assert np.allclose(mtr_3_r, ref_mtr)

    def test_pse_inv_with_np(self):
        # np.random.seed(100)
        for _ in range(5):
            shape = np.random.randint(1, 10, 2)
            target_mt = np.random.rand(*shape)
            np_ref = np.linalg.pinv(target_mt)
            pse_inv_res = pse_inv(target_mt)
            assert np.allclose(pse_inv_res, np_ref)

    def test_pse_inv_close(self):
        # np.random.seed(200)
        for _ in range(5):
            shape = np.random.randint(1, 20, 2)
            rand_mt = np.random.rand(*shape)
            inv_mt = pse_inv(rand_mt)
            diff = np.dot(np.dot(rand_mt, inv_mt), rand_mt) - rand_mt
            assert np.allclose(np.linalg.norm(diff), 0)

    def test_maximum_overlap(self):
        # oned case
        array_a = np.random.rand(5).reshape(5, 1)
        array_a /= np.linalg.norm(array_a)
        array_b = np.random.rand(5).reshape(5, 1)
        array_b /= np.linalg.norm(array_b)
        tf_mtr = maximum_overlap(array_a, array_b)
        new_b = np.dot(tf_mtr, array_b)
        assert_allclose(array_a, new_b)
        # nd case
        rand_c = np.random.rand(5, 5)
        array_c = np.linalg.eigh(np.dot(rand_c, rand_c.T))[1]
        array_c /= np.linalg.norm(array_c, axis=0)
        rand_d = np.random.rand(5, 5)
        array_d = np.linalg.eigh(np.dot(rand_d, rand_d.T))[1]
        array_d /= np.linalg.norm(array_d, axis=0)
        tf_mtr = maximum_overlap(array_c, array_d)
        new_d = np.dot(tf_mtr, array_d)
        assert_allclose(array_c, new_d)

    def test_procrustes(self):
        np.random.seed(101)
        for _ in range(5):
            a = np.random.rand(3, 2)
            n_a, s_a, m_a = np.linalg.svd(a)
            a_ref = n_a[:, :2]

            b = np.random.rand(3, 2)
            n_b, s_b, m_b = np.linalg.svd(b)
            b_ref = n_b[:, :2]
            result = procrustes(a_ref, b_ref)
            assert np.allclose(result, b_ref)

        for _ in range(5):
            a = np.random.rand(6, 4)
            n_a, s_a, m_a = np.linalg.svd(a)
            a_ref = n_a[:, :4]

            b = np.random.rand(6, 4)
            n_b, s_b, m_b = np.linalg.svd(b)
            b_ref = n_b[:, :4]
            result = procrustes(a_ref, b_ref)
            assert np.allclose(result, b_ref)
