import numpy as np
import horton as ht

from copy import deepcopy

class TS_Treat(object):

    def __init__(self, ts_state, key_ic_number):
        self.ts_state = ts_state
        self.key_ic = key_ic_number
        self.v_matrix = None
        self._old_v_matrix = None

    def _matrix_a_eigen(self):
        """calculate eigenvalue of b_matrix, select 3n-5 to form the a matrix

        Returns:
            numpy.array: shape(n, 3n-5), A matrix
        """
        b_matrix = deepcopy(self.ts_state.b_matrix)
        u, s, v = np.linalg.svd(b_matrix, full_matrices=False) #u.shape = (n, 3N)
        ic_len = len(self.ts_state.ic)
        a_matrix = np.zeros((ic_len, self.ts_state._dof), float)
        counter = 0
        for i in range(len(s)):
            if s[i] < 0.01:
                continue
            a_matrix[:,counter] = u[:, i]
            counter += 1
            if counter >= (self.ts_state._dof):
                break
        return a_matrix

    def _projection(self):
        """project perturbation on each key internal coordinates into relizable internal coordinates

        Returns:
            numpy.array: shape(n, R)
        """
        b_matrix = self.ts_state.b_matrix
        b_pinv = np.linalg.pinv(b_matrix)
        prj_matrix = np.dot(b_matrix, b_pinv)
        ic_len = len(self.ts_state.ic)
        ic_keyic_len = self.key_ic
        e_perturb = np.zeros((ic_len, ic_keyic_len), float)
        identity_matrix = np.identity(ic_keyic_len)
        e_perturb[:ic_keyic_len,:] = identity_matrix
        b_perturb = np.dot(prj_matrix, e_perturb)
        return b_perturb

    @staticmethod
    def gram_ortho(vectors, transpose=False):
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
        vec_len = len(vectors[0]) # numnber of columns(arrays)
        # gram = np.zeros((vec_len, vec_len), float)
        # for row in range(vec_len):
        #     for column in range(vec_len):
        #         gram[row][column] = np.dot(vectors[row], vectors[column])
        gram = np.dot(vectors.T, vectors)
        eig_value, eig_vector = np.linalg.eig(gram)
        basisset = np.zeros((vec_len, vec_len), float)
        counter = 0
        for i in range(vec_len):
            if eig_value[i] > 0.01:
                basisset[:,counter] = eig_vector[:, i]
                counter += 1
        return basisset[:,:counter] # numpy.array, shape(, counter)

    def _deloc_reduce_ic(self):
        """orthogonize perturbation, calculate reduced internal coordinates for key ic
        
        Returns:
            numpy.array: reduced internal coordinates
        """
        b_perturb = self._projection()
        basisset = self.gram_ortho(b_perturb)
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
        reduced_space_2 = np.dot(reduced_space_1, a_matrix)
        nonreduced_space = a_matrix - reduced_space_2
        non_reduced_num = self.ts_state._dof - self.key_ic
        return nonreduced_space[:,:non_reduced_num]

    def _nonreduce_ic(self):
        """calculate nonreduce internal coordinates
        
        Returns:
            numpy.array: nonreduced internal coordinates
        """
        d_vectors = self._deloc_non_reduce_ic()
        # print "d_vectors",d_vectors.shape
        basisset = self.gram_ortho(d_vectors)
        # print "basis", basisset.shape
        nonreduce_ic = np.dot(d_vectors, basisset)
        return nonreduce_ic

    def get_v_basis(self):
        """get 3n-5 nonredundant internal coordinates
        
        Returns:
            numpy.array: nonredundant internal coordinates
        """
        reduced = self._deloc_reduce_ic()
        non_reduced = self._nonreduce_ic()
        self._old_v_matrix = self.v_matrix
        self.v_matrix = np.hstack((reduced, non_reduced))

    def procruste_q(self):
        """procruste process to find the most overlapped V matrix
        
        Returns:
            numpy.array: shape(3N - 5 or 3N - 6, n), most overlapped V matrx
        """
        s = np.dot(self.v_matrix.T, self._old_v_matrix)
        u, sigma, w = np.linalg.svd(s)
        q_min = np.dot(u, w)
        max_v = np.dot(self.v_matrix, q_min)
        return max_v 
