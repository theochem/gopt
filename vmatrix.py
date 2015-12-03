import numpy as np

from copy import deepcopy

class Vmatrix(object):

    def __inti__(self, structure, key_ic_number):
        self.structure = deepcopy(structure)
        self.key_number = key_ic_number
        self.ic_len = len(self.structure.ic)

    def _matrix_a_eigen(self):
        """calculate eigenvalue of b_matrix, select 3n-5 to form the a matrix

        Returns:
            numpy.array: shape(3N - 5, n), A matrix
        """
        matrix_space = np.dot(self.structure.b_matrix, self.structure.b_matrix.transpose())
        eig_value, eig_vector = np.linalg.eig(matrix_space)
        a_matrix = np.zeros((self.key_number, ic_len), float)
        counter = 0
        for i in len(eig_value):
            if eig_value[i] < 0.01:
                continue
            a_matrix[counter] = eig_value[:, i]
            counter += 1
            if counter >= (self.key_number):
                break
        return a_matrix

    def _projection(self):
        """project perturbation on each key internal coordinates into relizable internal coordinates

        Returns:
            numpy.array: shape(n, R)
        """
        b_matrix = self.structure.b_matrix
        b_pinv = np.linalg.pinv(b_matrix)
        prj_matrix = np.dot(b_matrix, b_pinv)
        ic_len = len(self.structure.ic)
        ic_keyic_len = self.key_number
        e_perturb = np.identity(ic_keyic_len)
        b_perturb = np.dot(prj_matrix, e_perturb)
        return b_perturb

    @staticmethod
    def _gram_ortho(vectors, transpose=False):
        """grammian orthogonal treatment, to orthogonize the row space
        
        Args:
            vectors (numpy.array): a set of vectors to be orthogonized
            transpose (bool, optional): if the vactor span a column space, true
                to transpose it into row space
        
        Returns:
            numpy.array: orthogonlized vectors set. span in row space.
        """
        if transpose:
            vectors = vectors.T
        vec_len = len(vectors)
        gram = np.zeros((vec_len, vec_len), float)
        for row in range(vec_len):
            for column in range(vec_len):
                gram[row][column] = np.dot(vectors[row], vectors[column])
        eig_value, eig_vector = np.linalg.eig(gram)
        basisset = np.zeros((vec_len, vec_len), float)
        counter = 0
        for i in range(vec_len):
            if eig_value[i] > 0.01:
                basisset[counter] = eig_value[:, i]
                counter += 1
        return basisset[:counter]

    def _deloc_reduce_ic(self):
        """orthogonize perturbation, calculate reduced internal coordinates for key ic
        
        Returns:
            numpy.array: reduced internal coordinates
        """
        b_perturb = self._projection()
        basisset = self._gram_ortho(b_perturb)
        reduced_ic = np.dot(b_perturb, basisset)
        return reduced_ic

    def _deloc_non_reduce_ic(self):
        """calculate nonreduced_space by project a_matrix to nonspace of reduced space
        
        Returns:
            numpy.array: nonreduced vectors to form nonreduced space
        """
        a_matrix = self._matrix_a_eigen()
        v_reduce = self._deloc_reduce_ic()
        reduced_space_1 = np.dot(v_reduce, v_reduce.T)
        reduced_space_2 = np.dot(non_reduced_space_1, a_matrix.T)
        nonreduced_space = a_matrix - reduced_space_2
        return nonreduced_space

    def _nonreduce_ic(self):
        """calculate nonreduce internal coordinates
        
        Returns:
            numpy.array: nonreduced internal coordinates
        """
        d_vectors = self._deloc_non_reduce_ic()
        basisset = self._gram_ortho(d_vectors)
        nonreduce_ic = np.dot(d_vectors, basisset)
        return nonreduce_ic

    def get_v_basis(self):
        """get 3n-5 nonredundant internal coordinates
        
        Returns:
            numpy.array: nonredundant internal coordinates
        """
        reduced = self._deloc_reduce_ic()
        non_reduced = self._nonreduce_ic()
        return np.vstack((reduced, non_reduced))