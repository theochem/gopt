import numpy as np

from unittest import TestCase
from saddle.optimizer.hessian_modify import (modify_hessian,
                                             modify_hessian_with_pos_defi)


class TestHessianModify(TestCase):
    def setUp(self):
        np.random.seed(16)
        matrix = np.random.rand(5, 5)
        self.herm = np.dot(matrix.T, matrix)

    def test_all_pos(self):
        _, vectors = np.linalg.eigh(self.herm)
        values = np.arange(-1, -6, -1)
        new_matrix = np.dot(np.dot(vectors, np.diag(values)), vectors.T)
        modified_m = modify_hessian(new_matrix, neg_num=0, key_ic=0)
        values = np.linalg.eigh(modified_m)[0]
        assert np.allclose(values, [0.05, 0.05, 0.05, 0.05, 0.05])

        values = np.arange(2, -3, -1)
        new_matrix = np.dot(np.dot(vectors, np.diag(values)), vectors.T)
        modified_m = modify_hessian(new_matrix, neg_num=0, key_ic=0)
        values = np.linalg.eigh(modified_m)[0]
        assert np.allclose(values, [0.05, 0.05, 0.05, 1, 2])

        values = np.arange(0.1, 1, 0.2)
        new_matrix = np.dot(np.dot(vectors, np.diag(values)), vectors.T)
        modified_m = modify_hessian(new_matrix, neg_num=0, key_ic=0)
        values = np.linalg.eigh(modified_m)[0]
        assert np.allclose(values, [0.1, 0.3, 0.5, 0.7, 0.9])

        values = np.arange(0.01, 0.1, 0.02)
        new_matrix = np.dot(np.dot(vectors, np.diag(values)), vectors.T)
        modified_m = modify_hessian(new_matrix, neg_num=0, key_ic=0)
        values = np.linalg.eigh(modified_m)[0]
        assert np.allclose(values, [0.05, 0.05, 0.05, 0.07, 0.09])

    def test_one_neg(self):
        _, vectors = np.linalg.eigh(self.herm)
        values = np.arange(1, 6)
        new_matrix = np.dot(np.dot(vectors, np.diag(values)), vectors.T)
        modified_m = modify_hessian(new_matrix, neg_num=1, key_ic=2)
        values, n_vectors = np.linalg.eigh(modified_m)
        assert np.allclose(n_vectors[:, 0]**2,
                           np.array([
                               0.69293252, 0.38870599, -0.41219818,
                               -0.24030381, -0.37563137
                           ])**2)
        assert np.allclose(values, [-0.05, 2, 3, 4, 5])

        modified_m = modify_hessian(new_matrix, neg_num=1, key_ic=4)
        values, n_vectors = np.linalg.eigh(modified_m)
        assert np.allclose(n_vectors[:, 0]**2,
                           np.array([
                               0.29402663, -0.24170677, -0.35568675,
                               0.84128791, 0.14438752
                           ])**2)
        assert np.allclose(values, [-0.05, 1, 3, 4, 5])

        values = np.array([1, -1, -2, 3, 4])
        new_matrix = np.dot(np.dot(vectors, np.diag(values)), vectors.T)
        modified_m = modify_hessian(new_matrix, neg_num=1, key_ic=1)
        values, n_vectors = np.linalg.eigh(modified_m)
        assert np.allclose(n_vectors[:, 0]**2,
                           np.array([
                               0.41754632, -0.46959165, -0.08110722,
                               -0.42598055, 0.64583353
                           ])**2)
        assert np.allclose(values, [-2, 0.05, 1, 3, 4])

        values = np.array([-0.01, 2, 3, 4, 5])
        new_matrix = np.dot(np.dot(vectors, np.diag(values)), vectors.T)
        modified_m = modify_hessian(new_matrix, neg_num=1, key_ic=2)
        values, n_vectors = np.linalg.eigh(modified_m)
        assert np.allclose(n_vectors[:, 0]**2,
                           np.array([
                               0.69293252, 0.38870599, -0.41219818,
                               -0.24030381, -0.37563137
                           ])**2)
        assert np.allclose(values, [-0.05, 2, 3, 4, 5])

    def test_multi_neg(self):
        _, vectors = np.linalg.eigh(self.herm)
        values = np.array([-1, -2, -3, -4, -5])
        new_matrix = np.dot(np.dot(vectors, np.diag(values)), vectors.T)
        modified_m = modify_hessian(new_matrix, neg_num=2, key_ic=2)
        values, n_vectors = np.linalg.eigh(modified_m)
        assert np.allclose(n_vectors[:, 1]**2,
                           np.array([
                               0.69293252, 0.38870599, -0.41219818,
                               -0.24030381, -0.37563137
                           ])**2)
        assert np.allclose(n_vectors[:, 0]**2,
                           np.array([
                               0.26951539, 0.63812477, 0.48589596, 0.21054014,
                               0.48962863
                           ])**2)
        assert np.allclose(values, [-5, -1, 0.05, 0.05, 0.05])

        values = np.array([1, 2, -2, 0.001, 0.5])
        new_matrix = np.dot(np.dot(vectors, np.diag(values)), vectors.T)
        modified_m = modify_hessian(new_matrix, neg_num=2, key_ic=4)
        values, n_vectors = np.linalg.eigh(modified_m)
        assert np.allclose(n_vectors[:, 0]**2,
                           np.array([
                               0.41754632, -0.46959165, -0.08110722,
                               -0.42598055, 0.64583353
                           ])**2)
        assert np.allclose(n_vectors[:, 1]**2,
                           np.array([
                               0.29402663, -0.24170677, -0.35568675,
                               0.84128791, 0.14438752
                           ])**2)
        assert np.allclose(values, [-2, -0.05, 0.05, 0.5, 1])

    def test_mody_hes_with_pos_defi(self):
        init_matrix = np.array(
            [[0.11692545, -0.10407751, 0.22927565, 0.71404804, 0.48594549], [
                -0.10407751, -0.5162238, 1.35542663, 0.35645273, -0.37051617
            ], [0.22927565, 1.35542663, -1.42547168, 0.57380311, 0.26283181],
             [0.71404804, 0.35645273, 0.57380311, -0.20644067, 1.01414279],
             [0.48594549, -0.37051617, 0.26283181, 1.01414279, -0.4687893]])
        result_mat = modify_hessian_with_pos_defi(init_matrix, 1, 2)
        values = np.linalg.eigh(result_mat)[0]
        assert np.allclose(
            values, [-0.45498585, 0.05, 0.05726128, 1.11660122, 1.55024927])
        result_mat = modify_hessian_with_pos_defi(init_matrix, 2, 2)
        values = np.linalg.eigh(result_mat)[0]
        assert np.allclose(values, [
            -1.72157269, -0.45498585, 0.05726128, 1.11660122, 1.55024927
        ])

        np.random.seed(133)
        rand_mat = np.random.rand(5, 5)
        init_matrix = np.dot(rand_mat.T, rand_mat)
        result_mat = modify_hessian_with_pos_defi(init_matrix, 1, 3)
        values = np.linalg.eigh(result_mat)[0]
        np.allclose(values, [
            -0.05, 6.70424398e-02, 1.74207191e-01, 6.29035498e-01,
            5.86943965e+00
        ])
