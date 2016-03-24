import numpy as np
import math

from trust_radius import default_trust_radius
from copy import deepcopy
# from saddle.saddlepoint import SaddlePoint


class TrialOptimizer(object):
    """Optimizer class to optimize geometry and relative information to saddle point

    Attributes:
        points (list): a list of SaddlePoint instance
        trm_class (dict, class property): trust_radius method chooser, availabel method keyword "default"
    """

    def __init__(self):
        self.points = []
        self._trust_radius = None
        # self.parents=[]

    def _update_hessian_finite_difference(self, index, perturb=0.001):
        point = self.points[index]
        for i in range(point.key_ic_number):
            e_pert = np.zeros(point.ts_state._dof)
            e_pert[i] = 1.
            new_ts_state = point.reference.obtain_new_cc_with_new_delta_v(
                e_pert)
            new_point = new_ts_state.create_a_saddle_point()
            pt1 = (new_point.g_matrix - point.g_matrix) / perturb
            pt2 = np.dot(point.reference.v_matrix.T, np.linalg.pinv(
                point.reference.ts_state.b_matrix))
            dv = (new_point.reference.v_matrix - point.reference.v_matrix) / perturb
            pt3 = np.dot(point.reference.ts_state.b_matrix.T, np.dot(dv, point.g_matrix))
            db = (new_point.reference.ts_state.b_matrix - point.reference.ts_state.b_matrix) / perturb
            pt4 = np.dot(db.T, point.reference.ts_state.ic_gradient)
            point.hessian[:, i] = pt1 - np.dot(pt2, (pt3 + pt4))

    def set_trust_radius_method(self, **kwmethod): #checked
        """select keyword args to implement different trust radius methods

        Args:
            **kwmethod: 
                method: default value is "default", other choices:
                parameter: default value is None

        Returns:
            TYPE: Description
        """
        method = kwmethod.pop("method", "default")
        parameter = kwmethod.pop("parameter", None)
        trm_method = TrialOptimizer.trm_class[method]
        if parameter:
            self._trust_radius = trm_method(parameter)
        else:
            self._trust_radius = trm_method()

    def initialize_trm_for_point_with_index(self, index): #checked
        """initilize point with selected trust radius method

        Args:
            index (int): index of point in self.points attribute

        """
        point = self.points[index]
        self._trust_radius.initilize_point(point)

    def update_trm_for_point_with_index(self, index, method):
        """update trust radius method for a certain point

        Args:
            index (int): the index of point in attribute self.points
            method (string): string name to select trust radius update method
                "energy" for energy based trust radius update method
                "gradient" for gradient based trust radius update method

        """
        point = self.points[index]
        pre_point = self.points[index - 1]
        update_method = self._trust_radius.update_trust_radius(method)
        update_method(point, pre_point)

    @property
    def latest_index(self):
        """return the index of latest point

        Returns:
            int: return the index
        """
        return len(self.points) - 1

    def add_a_point(self, point): #checked
        """add a point to self.points attribute

        Args:
            point (point instance): a new point instance with updated information
        """
        self.points.append(point)

    def update_hessian_for_a_point(self, index, **kwmethod): 
        """update hessian for a certain point

        Args:
            index (int): the index of point in self.points
            **kwmethod: keyword args
                method: the method for updating hessian

        Raises:
            TypeError: unexpected **kwargs

        """
        method = kwmethod.pop("method")
        if kwmethod:
            raise TypeError('Unexpected **kwargs: {}'.format(kwmethod))
        point = self.points[index]
        pre_point = self.points[index - 1]
        if point.hessian:
            print("exists, quit updating")
        else:
            method = hessian_update_method[method]
            point.hessian = hessian_update_method[
                method]  # function need to be added here

    def update_hessian_for_latest_point(self, **kwmethod):
        """update hessian for the latest point

        Args:
            **kwmethod: keyword args
                method: the method for updating hessian

        """
        self.update_hessian_for_a_point(self.latest_index, **kwmethod)

    def _test_necessity_for_finite_difference(self, index):
        point = self.points[index]
        pre_point = self.points[index - 1]
        for i in range(point.key_ic_number):
            # create a perturbation array
            e_pert = np.zeros(self.point.ts_state._dof)
            e_pert[i] = 1
            if point.g_matrix[i] > np.linalg.norm(point.g_matrix) / math.sqrt(point.reference.ts_state._dof) and \
                    np.linalg.norm(np.dot(point.h_matrix, e_pert) - np.dot(pre_point.h_matrix, e_pert)) > \
                    1.0 * np.linalg.norm(np.dot(pre_point.h_matrix, e_pert)):
                return False
        # return True for no need to update through finite difference,
        # otherwise return False.
        return True

    def tweak_hessian_for_a_point(self, index): #checked
        """tweak the hessian for a point in self.points

        Args:
            index (int): the index of point
        """
        point = self.points[index]
        self._tweak_hessian(point)

    def tweak_hessian_for_latest_point(self): #checked
        """tweak the hessian for the latest point

        """
        point = self.points[self.latest_index]
        self._tweak_hessian(point)

    def _tweak_hessian(self, point):
        """the function to tweak the hessian

        Args:
            point (TYPE): Description

        Returns:
            TYPE: Description
        """
        point._diagnolize_h_matrix()  # diagnolize hessian matrix
        # modify the eigenvalue of hessian to make sure there is only one
        # negative eigenvalue
        point._modify_h_matrix()
        point._reconstruct_hessian_matrix()  # reconstruct hessian matrix

    def find_stepsize_for_a_point(self, index, **kwmethod): #checked
        """find the proper stepsize for a certain point

        Args:
            index (int): the index of point in self.points
            **kwmethod: keyword args
                method: the stepsize determination method
                    choices: "TRIM" method, "RFO" method

        Raises:
            TypeError: Unexpected keyword args

        """
        point = self.points[index]
        method = kwmethod.pop("method")
        if kwmethod:
            raise TypeError('Unexpected **kwargs: {}'.format(kwmethod))
        if method == "TRIM":
            stepsize = point._trust_region_image_potential()
        elif method == "RFO":
            stepsize = point._rational_function_optimization()
        point.stepsize = stepsize

    def find_stepsize_for_latest_point(self, **kwmethod): #checked
        """find the proper stepsize for latest point

        Args:
            index (int): the index of point in self.points
            **kwmethod: keyword args
                method: the stepsize determination method
                    choices: "TRIM" method, "RFO" method

        """
        self.find_stepsize_for_a_point(self.latest_index, **kwmethod)

    def update_to_new_point_for_a_point(self, index): #chekced
        point = self.points[index]
        new_point = point.obtain_new_cc_with_new_delta_v(point.stepsize)
        return new_point

    def _check_new_point_satisfied(self, old_point, new_point):
        no1 = np.linalg.norm(old_point.ts_state.gradient_matrix)
        no2 = np.linalg.norm(new_point.ts_state.gradient_matrix)
        print "no1, no2", no1, no2
        if no2 > no1:
            return False
        return True

    def _change_trust_radius_step(self, index, multiplier):
        point = self.points[index]
        new_control = point.step_control * multiplier
        point.step_control = max(new_control, self._trust_radius.min)



    def update_to_new_point_for_latest_point(self): #checked
        return self.update_to_new_point_for_a_point(self.latest_index)

    def update_trust_radius_for_a_point(self, index, **kwmethod):
        """update the trust radius for a certain point

        Args:
            index (int): the index of point need to be update 
            **kwmethod: keyword args
                method: the method for update trust radius
                parameter: the parameter to use the corresponding method

        Raises:
            IndexError: the index of point is invalid
            TypeError: provide unexpected kwargs

        """
        if index == 0:
            raise IndexError("Cannot update trust radius method")
        point = self.points[index]
        pre_point = self.points[index - 1]
        method = kwmethod.pop('method')
        parameter = kwmethod.pop('method', None)
        if kwmethod:
            raise TypeError('Unexpected **kwargs: {}'.format(kwmethod))
        trust_radius_update_method = trm_update_method[method]
        if parameter:
            trust_radius_update_method(
                self._trust_radius, point, pre_point, parameter)
        else:
            trust_radius_update_method(self._trust_radius, point, pre_point)

    def update_trust_radius_latest_point(self, **kwmethod):
        """update the trust radius for a latest point
        
        Args:
            **kwmethod: keyword args
                method: the method for update trust radius
                parameter: the parameter to use the corresponding method
        
        Raises:
            IndexError: the index of point is invalid
            TypeError: provide unexpected kwargs

        """
        self.update_trust_radius_for_a_point(self.latest_index, **kwmethod)

    def _secant_condition(self, point, point_old):
        part1 = point.g_matrix - point_old.g_matrix
        part2 = np.dot(point.reference.v_matrix.T,
                       np.linalg.inv(point.reference.b_matrix))
        part3 = np.dot(np.dot(point.reference.b_matrix.T,
                              (point.reference.v_matrix - point_old.reference.v_matrix)), point.g_matrix)
        part4 = np.dot((point.reference.b_matrix -
                        point_old.reference.b_matrix).T, point.reference.g_matrix_q)
        secant_value = part1 - np.dot(part2, (part3 + part4))
        return secant_value



    # hessian_update_method = {
    #     'SR1' : TrialOptimizer._sr1_update_method,
    #     'PSB' : TrialOptimizer._psb_update_method,
    #     'BFGS' : TrialOptimizer._bfgs_update_method,
    #     'Bofill' : TrialOptimizer._bofill_update_method,
    # }

    @staticmethod
    def _sr1_update_method(point, point_old, secant_value):
        part1 = secant_value - np.dot(point_old.h_matrix, point_old.stepsize)
        part2 = point_old.stepsize
        half_result = np.dot(part1, part2)
        numerator = np.dot(half_result.T, half_result)
        denominator = np.linalg.norm(part1) ** 2 * np.linalg.norm(part2) ** 2
        result = numerator / denominator
        if result <= 1E-18:
            point.h_matrix = np.deepcopy(point_old.h_matrix)
        else:
            new_value = point_old.h_matrix + \
                np.dot(part1, part1.T) / np.dot(part1, point_old.stepsize)
            point.h_matrix = new_value

    @staticmethod
    def _psb_update_method(point, point_old, secant_value):
        part1 = secant_value - np.dot(point_old.h_matrix, point_old.stepsize)
        part2 = point_old.stepsize
        value1 = point_old.h_matrix
        value2 = (np.dot(part1, part2.T) + np.dot(part2, part1.T)) / \
            np.dot(part2.T, part2)
        value3 = np.dot(part2.T, part1) / (np.dot(part2.T, part2) ** 2)
        value4 = np.dot(part2, part2.T)
        new_value = value1 + value2 - np.dot(value3, value4)
        point.h_matrix = new_value

    @staticmethod
    def _bfgs_update_method(point, point_old, secant_value):
        part1 = p.dot(point_old.h_matrix, point_old.stepsize)
        part2 = point_old.stepsize
        value1 = point_old.h_matrix
        value2 = np.dot(secant_value, secant_value.T) / \
            np.dot(secant_value.T, part2)
        value3 = np.dot(part1, part1.T) / np.dot(part2.T, part1)
        new_value = value1 + value2 - value3
        point.h_matrix = new_value

    @staticmethod
    def _bofill_update_method(point, point_old, secant_value):
        part1 = secant_value - p.dot(point_old.h_matrix, point_old.stepsize)
        part2 = point_old.stepsize
        norm = np.linalg.norm
        psi = 1 - norm(np.dot(part2, part1)) ** 2 / \
            np.dot(norm(part2) ** 2, norm(part1) ** 2)
        result1 = TrialOptimizer._sr1_update_method(
            point, point_old, secant_value)
        result2 = TrialOptimizer._psb_update_method(
            point, point_old, secant_value)
        new_value = (1. - psi) * result1 + psi * result2
        point.h_matrix = new_value

    trm_class = {
        "default" : default_trust_radius,
    }

    trm_update_method = {
        "energy" : default_trust_radius._energy_based_trust_radius_method,
        "gradient" : default_trust_radius._gradient_based_trust_radius_method,
    }

    hessian_update_method = {
        'SR1' : _sr1_update_method,
        'PSB' : _psb_update_method,
        'BFGS' : _bfgs_update_method,
        'Bofill' : _bofill_update_method,
    }


# class trust_radius(object):
#     pass


# class default_trust_radius(trust_radius):

#     def __init__(self, num_atoms):
#         self._max = math.sqrt(num_atoms)
#         self._min = 1. / 10 * math.sqrt(num_atoms)
#         # self._value = 0.35 * math.sqrt(num_atoms)

#     def initilize_point(self, point):
#         point.step_control = 0.35 * self.max

#     @property
#     def max(self):
#         return self._max

#     @property
#     def min(self):
#         return self._min

#     # def update_trust_radius(self, method):
#     #     if method == "energy":
#     #         return self._energy_based_trust_radius_method
#     #     elif method == "gradient":
#     #         return self._gradient_based_trust_radius_method

#     def _energy_based_trust_radius_method(self, point, pre_point):
#         delta_m = np.dot(pre_point.g_matrix, pre_point.stepsize) + 1. / 2 * np.dot(
#             np.dot(pre_point.stepsize.T, pre_point.h_matrix), pre_point.stepsize)  # approximated energy changes
#         delta_u = point.value - pre_point.value  # accurate energy changes
#         ratio_of_m_over_u = delta_m / delta_u
#         if ratio_of_m_over_u > 2. / 3 and ratio_of_m_over_u < 3. / 2:
#             point.step_control = min(
#                 max(2 * pre_point.step_control, self.min), self.max)
#         elif ratio_of_m_over_u > 1. / 3 and ratio_of_m_over_u < 3.:
#             point.step_control = max(pre_point.step_control, self.min)
#         else:
#             point.step_control = min(1. / 4 * pre_point.step_control, self.min)

#     def _gradient_based_trust_radius_method(self, point, pre_point, dimensions):
#         g_predict = pre_point.g_matrix + \
#             np.dot(pre_point.h_matrix, pre_point.stepsize)
#         norm = np.linalg.norm
#         ratio_rho = (norm(g_predict) - norm(pre_point.g_matrix)) / \
#             (norm(point.g_matrix) - norm(pre_point.g_matrix))
#         cos_ita = np.dot((g_predict - pre_point.g_matrix), (point.g_matrix - pre_point.g_matrix)) / \
#             np.dot(norm(g_predict - pre_point.g_matrix),
#                    norm(point.g_matrix, pre_point.g_matrix))
#         p_10 = math.sqrt(1.6424 / dimensions + 1.11 / dimensions ** 2)
#         p_40 = math.sqrt(0.064175 / dimensions + 0.0946 / dimensions ** 2)
#         if ratio_rho > 4. / 5 and ratio_rho < 5. / 4 and cos_ita > p_10:
#             point.step_control = min(
#                 max(2 * pre_point.step_control, self.min), self.max)
#         elif ratio_rho > 1. / 5 and ratio_rho < 6. and cos_ita > p_40:
#             point.step_control = max(pre_point.step, self.min)
#         else:
#             point.step_control = min(1. / 2 * pre_point.step, self.min)


'''EXAMPLE

a = TrialOptimizer()
a.add_a_point(point)
a.set_trust_radius_method(method=default)
a.initialize_trm_for_point_with_index(0)
a.update_hessian_for_latest_point(method=BFGS)
a.find_stepsize_for_a_point(method=TRIM, parameter=0.1)
new_point = point.update()
a.add_a_point(new_point)



'''
