from __future__ import division
import numpy as np
import horton as ht
import scipy.optimize as opt
from opt import ridders_solver

from copy import deepcopy, copy
# from saddle.saddlepoint import SaddlePoint


class TS_Treat(object):
    """class use to optimize the transition state, change its dimention to 3n-5

    Attributes:
        key_ic (int): number of key internal coordinates
        ts_state (ICTransformation object): a ICT instance contains the geometry information of
    a transition state
        v_matrix (numpy.array): the v matrix for optimization
    """

    def __init__(self, ts_state, key_ic_number):
        self.ts_state = ts_state
        self.key_ic = key_ic_number
        self.v_matrix = None
        self.v_gradient = None
        self.v_hessian = None
        self.step_control = None
        self.stepsize = None
        # self.advanced_info = {}

    def _matrix_a_eigen(self):
        """calculate eigenvalue of b_matrix, select 3n-5 to form the a matrix

        Returns:
            numpy.array: shape(n, 3n-5), A matrix
        """
        b_matrix = self.ts_state.b_matrix
        u, s, v = np.linalg.svd(
            b_matrix, full_matrices=False)  # u.shape = (n, 3N)
        ic_len = len(self.ts_state.ic)
        a_matrix = np.zeros((ic_len, self.ts_state.dof), float)
        counter = 0
        for i in range(len(s)):
            a_matrix[:, counter] = u[:, i]
            counter += 1
            if counter >= (self.ts_state.dof):
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
        assert ic_keyic_len > 0, "key internal coordinates should be more than 1"
        e_perturb = np.zeros((ic_len, ic_keyic_len), float)
        identity_matrix = np.identity(ic_keyic_len)
        e_perturb[:ic_keyic_len, :] = identity_matrix
        b_perturb = np.dot(prj_matrix, e_perturb)
        return b_perturb

    @staticmethod
    def gram_ortho(vectors, transpose=False):
        """gramian orthogonal treatment, to orthogonize the row space

        Args:
            vectors (numpy.array): a set of vectors to be orthogonized
            transpose (bool, optional): if the vactor span a column space, true
                to transpose it into row space

        Returns:
            numpy.array: orthogonlized vectors set. span in row space.
        """
        if transpose:
            vectors = vectors.T
        vec_len = len(vectors[0])  # numnber of columns(arrays)
        # gram = np.zeros((vec_len, vec_len), float)
        # for row in range(vec_len):
        #     for column in range(vec_len):
        #         gram[row][column] = np.dot(vectors[row], vectors[column])
        gram = np.dot(vectors.T, vectors)
        eig_value, eig_vector = np.linalg.eigh(gram)
        basisset = np.zeros((vec_len, vec_len), float)
        counter = 0
        # print "eigenvalue",eig_value, "\n",eig_vector
        for i in range(vec_len):
            if abs(eig_value[i]) > 1e-4:
                basisset[:, counter] = eig_vector[:, i]
                counter += 1
        return basisset[:, :counter]  # numpy.array, shape(, counter)

    def _new_deloc_reduce_ic(self): # can also be done by svd
        b_perturb = self._projection()
        reduced_ic = self.gram_ortho(b_perturb, True)
        return reduced_ic


    def _deloc_reduce_ic(self):
        """orthogonize perturbation, calculate reduced internal coordinates for key ic

        Returns:
            numpy.array: reduced internal coordinates
        """
        b_perturb = self._projection()
        basisset = self.gram_ortho(b_perturb)
        reduced_ic = np.dot(b_perturb, basisset)
        for i in range(len(reduced_ic[0])):
            reduced_ic[:, i] /= np.linalg.norm(reduced_ic[:, i])
        return reduced_ic

    def _null_space_of_reduce_space(self):
        a_matrix = self._matrix_a_eigen()
        v_reduce = self._new_deloc_reduce_ic()
        perjection_space = np.dot(v_reduce, v_reduce.T)
        return a_matrix - np.dot(perjection_space, a_matrix)

    def _new_deloc_nonreduce_ic(self):
        d_vectors = self._null_space_of_reduce_space()
        nonreduced_ic = self.gram_ortho(d_vectors, True)
        return nonreduced_ic

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
        # non_reduced_num = self.ts_state.dof - self.key_ic
        return nonreduced_space[:, :]

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
        for i in range(len(nonreduce_ic[0])):
            nonreduce_ic[:, i] /= np.linalg.norm(nonreduce_ic[:, i])
        return nonreduce_ic

    def get_new_v_basis(self):
        reduced_ic = self._new_deloc_reduce_ic()
        nonreduced_ic = self._new_deloc_nonreduce_ic()
        self.v_matrix = np.hstack((reduced_ic, nonreduced_ic))

    def get_v_basis(self):
        """get 3n-5 nonredundant internal coordinates

        Returns:
            numpy.array: nonredundant internal coordinates
        """
        reduced = self._deloc_reduce_ic()
        non_reduced = self._nonreduce_ic()
        self._old_v_matrix = self.v_matrix
        self.v_matrix = np.hstack((reduced, non_reduced))

    def obtain_new_cc_with_new_delta_v(self, delta_v, method, hessian=False, **kwargs):
        """calculate the change of internal coordinates \delta q according to the
        change of the change of V coordinates \delta v.

        Args:
            delta_v (numpy.array): the change of V coordinates, \delta v

        """
        delta_q = np.dot(self.v_matrix, delta_v)
        new_ts_state = deepcopy(self)  # deepcopy self
        new_ts_state.v_matrix = None
        new_ts_state.v_gradient = None
        new_ts_state.v_hessian = None
        # new_ts_state.advanced_info = {}
        new_ts_state.stepsize = None
        new_ts_state.ts_state.use_delta_ic_to_calculate_new_cc(delta_q)
        if hessian:
            new_ts_state.ts_state.get_energy_gradient_hessian(method, **kwargs)
            new_ts_state.get_v_basis()
            new_ts_state.get_v_gradient_hessian()
        else:
            new_ts_state.ts_state.get_energy_gradient(method, **kwargs)
            new_ts_state.get_v_basis()
            new_ts_state.get_v_gradient()
        return new_ts_state

    def get_v_gradient(self):
        q_g = self.ts_state.ic_gradient
        self.v_gradient = np.dot(self.v_matrix.T, q_g)

    def get_v_hessian(self):
        q_h = self.ts_state.ic_hessian
        self.v_hessian = np.dot(np.dot(self.v_matrix.T, q_h), self.v_matrix)

    def get_v_gradient_hessian(self):
        self.get_v_gradient()
        self.get_v_hessian()

    def set_ic_gradient(self):
        self.ts_state.ic_gradient = np.dot(self.v_matrix, self.v_gradient)

    def set_ic_hessian(self):
        self.ts_state.ic_hessian = np.dot(
            np.dot(self.v_matrix, self.v_hessian), self.v_matrix.T)

    def set_ic_x_gradient(self):
        self.set_ic_gradient()
        self.ts_state.gradient_ic_to_x()

    def set_ic_x_hessian(self):
        self.set_ic_hessian()
        self.ts_state.hessian_ic_to_x()

    def _tweak_hessian_nonreduced(self, negative=0):
        hessian = self.v_hessian.copy()
        nonreduced = hessian[self.key_ic:, self.key_ic:]
        w_r, v_r = np.linalg.eigh(nonreduced)
        w_r[w_r < 0.] = 0.
        hessian[self.key_ic:, self.key_ic:] = np.dot(v, np.dot(np.diag(w), v.T))
        return hessian

    def _tweak_hessian_eigenvalues(self, negative=0, threshold=0.005):
        self.v_hessian = self._tweak_hessian_nonreduced(negative) #change the hessian
        eigenvalues, eigenvectors = np.linalg.eigh(self.v_hessian)
        negative_eig = np.sum(eigenvalues < 0)
        if negative_eig == negative: # if the negative eigenvalues equal to indended one
            negative_index = np.arange(negative_eig)
            positive_index = np.arange(negative_eig, len(eigenvalues))

        elif negative_eig < negative:
            diff = negative - negative_eig
            fraction = np.sum(np.square(eigenvectors[:self.key_ic, negative_eig:]), axis=0)
            ngt_extra_index = fraction.argsort()[::-1][:diff] + negative_eig
            negative_index = np.append(np.arange(negative_eig), ngt_extra_index)
            positive_index = fraction.argsort()[::-1][diff:] + negative_eig

        elif negative_eig > negative:
            diff = negative_eig - negative
            fraction = np.sum(np.square(eigenvectors[:self.key_ic, :negative_eig]), axis=0)
            negative_index = fraction.argsort()[::-1][:negative]
            pst_extra_index = fraction.argsort()[::-1][negative:]
            positive_index = np.append(pst_extra_index, np.arange(negative_eig, len(self.eigenvalues)))

        ngt_e_vl = eigenvalues[negative_index] # negative eigenvalues
        pst_e_vl = eigenvalues[positive_index] # positive eigenvalues
        ngt_e_vl[ngt_e_vl > -threshold] = -threshold # change the nagetive eigenvalues to <= -threshold
        pst_e_vl[pst_e_vl < threshold] = threshold # change the positive eigenvalues to >= threshold
        eigenvalues[negative_index] = ngt_e_vl
        eigenvalues[positive_index] = pst_e_vl
        new_hessian = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), eigenvectors.T))
        return new_hessian

    def _trust_region_image_potential(self, limit, negative=0):
        init_step = -np.dot(np.linalg.pinv(self.v_hessian), self.v_gradient)
        if np.linalg.norm(init_step) <= limit:
            return init_step
        eigenvalues, eigenvectors = np.linalg.eigh(self.v_hessian)
        max_w = max(eigenvalues)
        def func_step(value):
            e_v_copy = eigenvalues.copy()
            e_v_copy[:negative] = e_v_copy[:negative] - value
            e_v_copy[negative:] = e_v_copy[negative:] + value
            new_hessian_inv = np.dot(v, np.dot(np.diag(1. / e_v_copy), v.T))
            return -np.dot(new_hessian_inv, self.v_gradient)

        def func_value(value):
            step = func_step(value)
            return np.linalg.norm(step) - limit

        while func_value(max_w) >= 0:
            max_w *= 2

        result = ridders_solver(func_value, 0, max_w)
        step = func_step(result)
        return step


    # def _diagnolize_h_matrix(self):
    #     """diagnolize hessian matrix if it is not none
    #     """
    #     w, v = np.linalg.eigh(
    #         self.v_hessian)  # w is the eigenvalues while v is the eigenvectors
    #     self.advanced_info["eigenvalues"] = w
    #     self.advanced_info["eigenvectors"] = v
    #
    # def _test_trust_radius_method(self):
    #     """test trust radius method, defined by Derrick myself.
    #     """
    #     origin_step = np.dot(np.linalg.pinv(self.v_hessian), np.v_gradient)
    #     length = np.linalg.norm(origin_step)
    #     new_step = origin_step / length * self.step_control
    #     return new_step
    #
    # def switch_eigens(self, one_index, the_other_index):
    #     """switch the eigen values and eigenvalues of two different indexes
    #
    #     Args:
    #         one_index (int): the one index to be switched
    #         the_other_index (int): the other index to be switched
    #     """
    #     # set temp eigenvalue and eigenvector
    #     eigenvalues = self.advanced_info["eigenvalues"]
    #     eigenvectors = self.advanced_info["eigenvectors"]
    #     eigenvalues[one_index], eigenvalues[the_other_index] = eigenvalues[
    #         the_other_index], eigenvalues[one_index]
    #     # assign the other index
    #     eigenvalues[one_index], eigenvalues[the_other_index] = eigenvalues[
    #         the_other_index], eigenvalues[one_index]
    #     eigenvectors[:, one_index], eigenvectors[:, the_other_index] = np.copy(
    #         eigenvectors[:, the_other_index]), np.copy(eigenvectors[:, one_index])
    #
    # def _modify_h_matrix(self, pos_thresh=0.005, neg_thresh=-0.005):
    #     """modify the eigenvalues of hessian matrix to make sure it has the right form
    #
    #     Args:
    #         pos_thresh (float, optional): the threshold for positive eigenvalues, default is 0.005
    #         neg_thresh (float, optional): the threshold for nagetive eigenvalues, default is -0.005
    #
    #     """
    #     total_number = self.ts_state.dof
    #     pos = 0
    #     neg = 0
    #     for i in range(total_number):  # here can be optimized, but i am lazy to do that
    #         if self.advanced_info["eigenvalues"][i] >= 0:
    #             pos += 1
    #         elif self.advanced_info["eigenvalues"][i] < 0:
    #             neg += 1
    #
    #     if neg == 1:  # ideal situation with only one negative eigenvalues
    #         for i in range(1, total_number):
    #             self.advanced_info["eigenvalues"][i] = max(
    #                 pos_thresh, self.advanced_info["eigenvalues"][i])
    #         self.advanced_info["eigenvalues"][0] = min(
    #             neg_thresh, self.advanced_info["eigenvalues"][0])
    #
    #     if neg > 1:  # method to select the most important negative eigenvalues
    #         fraction = 0  # initial value for fraction calculation
    #         label_flag = -1  # default flag, value -1 is just a symbol
    #         for i in range(neg):
    #             corresponding_eigenvector = self.advanced_info[
    #                 "eigenvectors"][:, i]
    #             temp_sum = 0
    #             for j in range(self.key_ic):
    #                 temp_sum += corresponding_eigenvector[j] ** 2
    #             if temp_sum > fraction:
    #                 fraction = temp_sum
    #                 label_flag = i
    #             # print "find max",i, temp_sum
    #         # switch the selected negative eigenvalue and vector to index 0
    #         if label_flag != 0:
    #             self.switch_eigens(0, label_flag)
    #             # print "negative eigen flag",label_flag
    #         for i in range(1, total_number):
    #             self.advanced_info["eigenvalues"][i] = max(
    #                 pos_thresh, self.advanced_info["eigenvalues"][i])
    #         self.advanced_info["eigenvalues"][0] = min(
    #             neg_thresh, self.advanced_info["eigenvalues"][0])
    #
    #     if neg == 0:  # choose the one more important eigenvalues to become negative
    #         # index for any eigenvectors that has more than 0.5 fraction in
    #         # reduced space
    #         lowest_eigenvalue = None
    #         label_flag = -1  # the same reason as above
    #         for i in range(total_number):
    #             corresponding_eigenvector = self.advanced_info[
    #                 "eigenvectors"][:, i]
    #             temp_sum = 0
    #             for j in range(self.key_ic):
    #                 temp_sum += corresponding_eigenvector[j]**2
    #             if temp_sum >= 0.5:
    #                 if self.advanced_info["eigenvalues"][i] < lowest_eigenvalue or lowest_eigenvalue == None:
    #                     lowest_eigenvalue = self.advanced_info[
    #                         "eigenvalues"][i]
    #                     label_flag = i
    #         if label_flag != 0:
    #             self.switch_eigens(0, label_flag)
    #         for i in range(1, total_number):
    #             self.advanced_info["eigenvalues"][i] = max(
    #                 pos_thresh, self.advanced_info["eigenvalues"][i])
    #         self.advanced_info["eigenvalues"][0] = min(
    #             neg_thresh, self.advanced_info["eigenvalues"][0])
    #
    # def _reconstruct_hessian_matrix(self):
    #     """reconstruct new hessian depends on the twieked hessian matrix
    #
    #     """
    #     eigenvalues = self.advanced_info["eigenvalues"]
    #     eigenvectors = self.advanced_info["eigenvectors"]
    #     self.v_hessian = np.dot(np.dot(eigenvectors, np.diag(
    #         eigenvalues)), eigenvectors.T)  # V W V.T
    #
    # def _trust_region_image_potential(self):
    #     """use TRIR method to find proper step under the control of trust radius method
    #
    #     Returns:
    #         numpy.array: the steps to be taken to update geometry
    #     """
    #     eigenvectors = self.advanced_info["eigenvectors"]
    #     eigenvalues = self.advanced_info["eigenvalues"]
    #     c_step = -np.dot(np.linalg.pinv(self.v_hessian), self.v_gradient)
    #     if np.linalg.norm(c_step) <= self.step_control:
    #         return c_step
    #     assert np.allclose(np.dot(np.dot(eigenvectors, np.diag(
    #         eigenvalues)), eigenvectors.T), self.v_hessian)
    #     eigenvalues, eigenvectors = np.linalg.eigh(self.v_hessian)
    #     max_w = max(eigenvalues)
    #
    #     def non_linear_value(lamda):  # define function for ridder method calculation
    #         w = eigenvalues.copy()
    #         v = eigenvectors
    #         print("lamda", lamda)
    #         w[:1] = w[:1] - lamda
    #         w[1:] = w[1:] + lamda
    #         new_hessian_inv = np.dot(v, np.dot(np.diag(1. / w), v.T))
    #         return -np.dot(new_hessian_inv, self.v_gradient)
    #
    #     def non_linear_func(lamda):
    #         s_value = non_linear_value(lamda)
    #         return np.linalg.norm(s_value) - self.step_control
    #
    #     while non_linear_func(max_w) >= 0:
    #         max_w *= 2
    #     result = ridders_solver(non_linear_func, 0, max_w)
    #     step = non_linear_value(result)
    #     return step
    #
    # def _rational_function_optimization(self):
    #     """use RFO method to find proper step under the control of trust radius method
    #
    #     Returns:
    #         numpy.array: the steps to be taken to update geometry
    #     """
    #     eigenvalues = self.advanced_info["eigenvalues"]
    #     eigenvectors = self.advanced_info["eigenvectors"]
    #     g_matrix = self.v_gradient
    #     # construct neg_matrix
    #     neg_matrix = np.zeros((2, 2), float)
    #     neg_matrix[0][0] = eigenvalues[0]
    #     neg_matrix[1][0] = np.dot(self.v_gradient.T, eigenvectors[:, 0])
    #     neg_matrix[0][1] = np.dot(eigenvectors[:, 0].T, self.v_gradient)
    #     eig_value_p, _ = np.linalg.eigh(neg_matrix)
    #     # construct neg_matrix
    #     pos_matrix = np.zeros((self.ts_state.dof, self.ts_state.dof), float)
    #     for i in range(1, self.ts_state.dof):
    #         pos_matrix[i - 1][i - 1] = eigenvalues[i]
    #         pos_matrix[self.ts_state.dof - 1][i -
    #                                           1] = np.dot(self.v_gradient.T, eigenvectors[:, i])
    #         pos_matrix[i - 1][self.ts_state.dof -
    #                           1] = np.dot(eigenvectors[:, i].T, self.v_gradient)
    #     eig_value_n, _ = np.linalg.eigh(pos_matrix)
    #
    #     def non_linear_value(lamda):  # define function for ridder method calculation
    #         eig_value_p_copy = deepcopy(eig_value_p)
    #         eig_value_p_copy[:-1] = eig_value_p_copy[: -1] * lamda
    #         lamda_p = max(eig_value_p_copy.flatten())
    #         if lamda == 0:
    #             lamda_p = 0
    #         part_1 = np.dot(eigenvectors[0].T,
    #                         g_matrix) / (eigenvalues[0] - lamda_p)
    #         part_1 = np.dot(part_1, eigenvectors[0])
    #         eig_value_n_copy = deepcopy(eig_value_n)
    #         eig_value_n_copy[:-1] = eig_value_n_copy[: -1] * lamda
    #         lamda_n = - min(eig_value_n_copy.flatten())
    #         if lamda == 0 or lamda_n < 0:
    #             lamda_n = 0
    #         part_2 = 0
    #         for i in range(1, self.ts_state.dof):
    #             temp_p2 = np.dot(
    #                 eigenvectors[i].T, g_matrix) / (eigenvalues[i] + lamda_n)
    #             temp_p2 = np.dot(temp_p2, eigenvectors[i])
    #             part_2 += temp_p2
    #         s_value = - part_1 - part_2
    #         return s_value
    #
    #     def non_linear_func(lamda):
    #         s_value = non_linear_value(lamda)
    #         return np.linalg.norm(s_value) - self.step_control
    #
    #     try_value = non_linear_func(0)
    #     if try_value < 0:
    #         return non_linear_value(0)
    #     try_lamda = 1
    #     while non_linear_func(try_lamda) > 0:
    #         try_lamda *= 2
    #     root_for_lamda = opt.ridder(non_linear_func, 0, try_lamda)
    #     return non_linear_value(root_for_lamda)
