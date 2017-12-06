import unittest
import numpy as np

from saddle.solver import ridders_solver, diagonalize
from saddle.errors import PositiveProductError

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
        func = lambda x: x**3 - x**2 - 2*x
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
