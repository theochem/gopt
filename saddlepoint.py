import numpy as np
import horton as ht
import scipy.optimize as opt
import math

from copy import deepcopy, copy

# this is the test class, will be implemented some point
# class BasicPoint(object):

#     def __init__(self, value, dimention_len, g_matrix, reference, h_matrix=None):
#         self.value = value
#         self.len = dimention_len
#         self.g_matrix = deepcopy(g_matrix)
#         self.h_matrix = deepcopy(h_matrix)
#         self.step_control = None
#         self.stepsize = None
#         self.reference = reference
#         self.advanced_info = {}

#     def update(self):
#         raise NotImplementedError("please rewrite it in subclass")


class SaddlePoint(object):
    """information treatment class for transition state, basic point class for optimization
    
    Attributes:
        advanced_info (dict): any advanced information need to be used to update point
        g_matrix (numpy.array): gradien matrix for optimization in cartesian coordinates
        h_matrix (numpy.array): hessian matrix for optimization, default is none.
        key_ic_number (int): number of key ic numbers
        len (int): number of internal coordinates
        step_control (float): the proper steplength for update steps
        stepsize (float): the proper step will be take
    """

    def __init__(self, length, g_matrix, vmatrix, reference, key_ic_number=0, h_matrix=[]):
        self.len = length
        self.g_matrix = deepcopy(g_matrix) # gradien matrix in internal coordinates
        # if h_matrix == []:
            # self.h_matrix = np.identity(self.len) # hessian matrix in internal coordinates
        self.h_matrix = deepcopy(h_matrix)
        self.advanced_info = {}
        self.key_ic_number = key_ic_number
        self.step_control = None
        self.reference = reference
        self.stepsize = None

    # def _test_necessity_for_finite_difference(self):
    #     for i in range(self.key_ic_number):
    #         if self.g_matrix[i] > np.linalg.norm(self.g_matrix) / math.sqrt(self.reference.ts_state._dof)

    def _diagnolize_h_matrix(self):
        """diagnolize hessian matrix if it is not none
        """
        w,v = np.linalg.eigh(self.h_matrix) # w is the eigenvalues while v is the eigenvectors
        self.advanced_info["eigenvalues"] = w
        self.advanced_info["eigenvectors"] = v

    # @staticmethod
    # def _change_sequence_eigen(eigenvalues, eigenvectors):
    #     length = len(eigenvalues)
    #     new_eigenvalues = [None] * length
    #     new_eigenvectors = np.zeros((length, length), float)
    #     for i in range(length):
    #         new_eigenvalues[i] = eigenvalues[length - 1 - i]
    #         new_eigenvectors[:,i] = eigenvectors[:, length - 1 - i]
    #     return new_eigenvalues, new_eigenvectors

    @staticmethod
    def switch_eigens(eigenvalues, eigenvectors, one_index, the_other_index):
        """switch the eigen values and eigenvalues of two different indexes
        
        Args:
            eigenvalues (numpy.array): numpy array of whole bunch of eigenvalues
            eigenvectors (numpy.array): numpy array of whole bunch of eigenvectors
            one_index (int): the one index to be switched
            the_other_index (int): the other index to be switched
        """
        # set temp eigenvalue and eigenvector
        temp_eigen_value = copy(eigenvalues[one_index])
        temp_eigen_vector = copy(eigenvectors[:, one_index])
        # assign the other index 
        eigenvalues[one_index] = copy(eigenvalues[the_other_index])
        eigenvectors[:, one_index] = copy(eigenvectors[:, the_other_index])
        # assign the temp value back to the other index
        eigenvalues[the_other_index] = copy(eigenvalues[one_index])
        eigenvectors[:, the_other_index] = copy(eigenvectors[:, one_index])

    def _modify_h_matrix(self, pos_thresh=0.005, neg_thresh=-0.005):
        """modify the eigenvalues of hessian matrix to make sure it has the right form
        
        Args:
            pos_thresh (float, optional): the threshold for positive eigenvalues, default is 0.005
            neg_thresh (float, optional): the threshold for nagetive eigenvalues, default is -0.005
        
        """
        total_number = self.len
        pos = 0
        neg = 0
        for i in range(total_number): # here can be optimized, but i am lazy to do that
            if self.advanced_info["eigenvalues"][i] >= 0:
                pos += 1
            elif self.advanced_info["eigenvalues"][i] < 0:
                neg += 1

        if neg == 1: # ideal situation with only one negative eigenvalues
            for i in range(1, total_number):
                self.advanced_info["eigenvalues"][i] = max(pos_thresh, self.advanced_info["eigenvalues"][i])
            self.advanced_info["eigenvalues"][0] = min(neg_thresh, self.advanced_info["eigenvalues"][0])

        if neg > 1: # method to select the most important negative eigenvalues
            fraction = 0 #initial value for fraction calculation
            label_flag = -1 #default flag, value -1 is just a symbol
            for i in range(neg):
                corresponding_eigenvector = self.advanced_info["eigenvectors"][:,i]
                temp_sum = 0
                for j in range(self.key_ic_number):
                    temp_sum += corresponding_eigenvector[j]**2
                if temp_sum > fraction:
                    fraction = temp_sum
                    label_flag = i
                print i, temp_sum
            #switch the selected negative eigenvalue and vector to index 0
            if label_flag != 0:
                SaddlePoint.switch_eigens(self.advanced_info["eigenvalues"], self.advanced_info["eigenvectors"], 0, label_flag)
                print label_flag
            for i in range(1, total_number):
                self.advanced_info["eigenvalues"][i] = max(pos_thresh, self.advanced_info["eigenvalues"][i])
            self.advanced_info["eigenvalues"][0] = min(neg_thresh, self.advanced_info["eigenvalues"][0])

        if neg == 0: # choose the one more important eigenvalues to become negative
            lowest_eigenvalue = None # index for any eigenvectors that has more than 0.5 fraction in reduced space
            label_flag = -1 # the same reason as above
            for i in range(total_number):
                corresponding_eigenvector = self.advanced_info["eigenvectors"][:,i]
                temp_sum = 0
                for j in range(self.key_ic_number):
                    temp_sum += corresponding_eigenvector[j]**2
                if temp_sum >= 0.5:
                    if self.eigenvalues[i] < lowest_eigenvalue or lowest_eigenvalue == None:
                        lowest_eigenvalue = self.eigenvalues[i]
                        label_flag = i
            if label_flag != 0:
                SaddlePoint.switch_eigens(self.advanced_info["eigenvalues"], self.advanced_info["eigenvectors"], 0, label_flag)
            for i in range(1, total_number):
                self.advanced_info["eigenvalues"][i] = max(pos_thresh, self.advanced_info["eigenvalues"][i])
            self.advanced_info["eigenvalues"][0] = min(neg_thresh, self.advanced_info["eigenvalues"][0])

    def _reconstruct_hessian_matrix(self):
        """reconstruct new hessian depends on the twieked hessian matrix

        """
        eigenvalues = self.advanced_info["eigenvalues"]
        eigenvectors = self.advanced_info["eigenvectors"]
        self.h_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), eigenvectors.T) # V W V.T

    def _trust_region_image_potential(self):
        """use TRIR method to find proper step under the control of trust radius method
        
        Returns:
            numpy.array: the steps to be taken to update geometry
        """
        eigenvectors = self.advanced_info["eigenvectors"]
        eigenvalues = self.advanced_info["eigenvalues"]
        g_matrix = self.g_matrix
        def non_linear_value(lamda): #define function for ridder method calculation
            part_1 = np.dot(eigenvectors[0].T, g_matrix) / (eigenvalues[0] - lamda)
            part_1 = np.dot(part_1, eigenvectors[0])
            part_2 = 0
            for i in range(1, self.len):
                temp_p2 = np.dot(eigenvectors[i].T, g_matrix) / (eigenvalues[i] + lamda)
                temp_p2 = np.dot(temp_p2, eigenvectors[i])
                part_2 += temp_p2
            s_value = - part_1 - part_2
            return s_value

        def non_linear_func(lamda):
            s_value = non_linear_value(lamda)
            return np.linalg.norm(s_value) - self.step_control
        try_value = non_linear_func(0)
        if try_value < 0:
            return non_linear_value(0) 
        try_eigen_value = eigenvalues[-1]
        while non_linear_func(try_eigen_value) > 0:
            try_eigen_value *= 2
        root_for_lamda = opt.ridder(non_linear_func, 0, try_eigen_value)
        return non_linear_value(root_for_lamda)

    def _rational_function_optimization(self):
        """use RFO method to find proper step under the control of trust radius method
        
        Returns:
            numpy.array: the steps to be taken to update geometry
        """
        eigenvectors = self.advanced_info["eigenvalues"]
        eigenvectors = self.advanced_info["eigenvectors"]
        #construct neg_matrix
        neg_matrix = np.zeros((2,2), float)
        neg_matrix[0][0] = eigenvalues[0]
        neg_matrix[1][0] = np.dot(self.g_matrix.T, eigenvectors[:,0])
        neg_matrix[0][1] = np.dot(eigenvectors[:,0].T, self.g_matrix)
        eig_value_p, _ = np.linalg.eigh(neg_matrix)
        #construct neg_matrix
        pos_matrix = np.zeros((self.len, self.len), float)
        for i in range(1, self.len):
            pos_matrix[i - 1][i - 1] = eigenvalues[i]
            pos_matrix[self.len - 1][i - 1] = np.dot(self.g_matrix.T, eigenvectors[:, i])
            pos_matrix[i - 1][self.len - 1] = np.dot(eigenvectors[:, i].T, self.g_matrix)
        eig_value_n, _ = np.linalg.eigh(pos_matrix)

        def non_linear_value(lamda): #define function for ridder method calculation
            eig_value_p_copy = deepcopy(eig_value_p)
            eig_value_p_copy[:-1] = eig_value_p_copy[: -1] * lamda
            lamda_p = max(eig_value_p_copy.flatten())
            if lamda == 0:
                lamda_p = 0
            part_1 = np.dot(eigenvectors[0].T, g_matrix) / (eigenvalues[0] - lamda_p)
            part_1 = np.dot(part_1, eigenvectors[0])
            eig_value_n_copy = deepcopy(eig_value_n)
            eig_value_n_copy[:-1] = eig_value_n_copy[: -1] * lamda
            lamda_n = - min(eig_value_n_copy.flatten())
            if lamda == 0 or lamda_n < 0:
                lamda_n = 0
            part_2 = 0
            for i in range(1, self.len):
                temp_p2 = np.dot(eigenvectors[i].T, g_matrix) / (eigenvalues[i] + lamda_n)
                temp_p2 = np.dot(temp_p2, eigenvectors[i])
                part_2 += temp_p2
            s_value = - part_1 - part_2
            return s_value

        def non_linear_func(lamda):
            s_value = non_linear_value(lamda)
            return np.linalg.norm(s_value) - self.step_control

        try_value = non_linear_func(0)
        if try_value < 0:
            return non_linear_value(0)
        try_lamda = 1
        while non_linear_func(try_lamda) > 0:
            try_lamda *= 2
        root_for_lamda = opt.ridder(non_linear_func, 0, try_lamda)
        return non_linear_value(root_for_lamda)

        #need to use ridder method to solve the function.

    # def update(self):
    #     new_ts_state = self.reference.obtain_new_cc_with_new_delta_v(self.stepsize)
    #     return newpoint = SaddlePoint(None, None) #need to be completed
