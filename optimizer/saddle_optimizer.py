import numpy as np
import math
import hessian_update as hu

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
        self._counter = 0

    def _update_hessian_finite_difference(self, index, perturb=0.001):
        """use finite difference method to update hessian if hessian matrix is 
        not provided or it performed terribly
        
        Args:
            index (int): index of point in self.points list.
            perturb (float, optional): the scale of perturbation added to each 
        dimention for calculation
        
        """
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

    # def update_trm_for_point_with_index(self, index, method):
    #     """update trust radius method for a certain point

    #     Args:
    #         index (int): the index of point in attribute self.points
    #         method (string): string name to select trust radius update method
    #             "energy" for energy based trust radius update method
    #             "gradient" for gradient based trust radius update method

    #     """
    #     point = self.points[index]
    #     pre_point = self.points[index - 1]
    #     update_method = self._trust_radius.update_trust_radius(method)
    #     update_method(point, pre_point)

    def veryfy_new_point_with_index(self, index, new_point):
        """to test new calculated point is competent to be keep, if not send back
        to recalculate a new point
        
        Args:
            index (int): the index of point in attribute self.points
            new_point (Ts_Treat): the point calculated through calculation
        
        Returns:
            bool: True if the point is in good condition, otherwise False
        """
        flag = True
        father_point = self.points[index]
        norm_new = np.linalg.norm(new_point.ts_state.gradient_matrix)
        norm_old = np.linalg.norm(father_point.ts_state.gradient_matrix)
        if norm_new > norm_old:
            flag = False
            father_point.step_control *= 0.25
            if father_point.step_control < 0.1 * self._trust_radius.min:
                father_point.step_control = self._trust_radius.min
        return flag

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
        self._counter_add()

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
        if point.v_hessian:
            print("exists, quit updating")
        else:
            method = TrialOptimizer.hessian_update_method[method]
            secand_value = self._secant_condition(point, pre_point)
            new_value = method(point, pre_point, secand_value)  # function need to be added here
            point.v_hessian = new_value
            print("finish updating")
        assert np.allclose(point.v_hessian, point.v_hessian.T) # make sure Hessian is symmetric

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
                    np.liSSRnalg.norm(np.dot(point.h_matrix, e_pert) - np.dot(pre_point.h_matrix, e_pert)) > \
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
        """update to a new point depent on the information of present point like
        hessian, trust radius method.
        
        Args:
            index (int): index of point in the attribute self.points
        
        Returns:
            Ts_Treat: the new point for further treatment and update
        """
        point = self.points[index]
        new_point = point.obtain_new_cc_with_new_delta_v(point.stepsize)
        return new_point


    def update_to_new_point_for_latest_point(self): #checked
        """update to a new point depent on the information of the latest point
        """
        return self.update_to_new_point_for_a_point(self.latest_index)


    # def _check_new_point_competent(self, old_point, new_point):
    #     """chech the ne
        
    #     Args:
    #         old_point (TYPE): Description
    #         new_point (TYPE): Description
        
    #     Returns:
    #         TYPE: Description
    #     """
    #     no1 = np.linalg.norm(old_point.ts_state.gradient_matrix)
    #     no2 = np.linalg.norm(new_point.ts_state.gradient_matrix)
    #     if no2 > no1:
    #         return False
    #     return True

    def _test_converge(self, point, old_point, method="gradient"):
        """test whether two point follow the rules of converg
        
        Args:
            point (Ts_Treat): newly calculated point
            old_point (Ts_Treat): the older point
            method (str, optional): criterion for determine converge
        
        Returns:
            Bool: True if it converge, otherwise False
        """
        gm = np.max(point.ts_state.gradient_matrix)
        condition_1  = (np.linalg.norm(gm) <= 3.e-4)
        condition_2 = (np.abs(point.ts_state.energy - old_point.ts_state.energy) <= 1e-6)
        delta_q = np.dot(point.v_matrix, point.stepsize)
        delta_x = np.dot(np.linalg.pinv(point.ts_state.b_matrix), delta_q)
        condition_3 = (np.max(np.abs(delta_x)) <= 3e-4)
        return condition_1 and (condition_2 or condition_3)

    def verify_convergence_for_a_point(self, index):
        """to test a point whether if achieve the convergence criterion
        
        Args:
            index (int): the index of point in self.points
        
        Returns:
            bool: True if it converged, otherwise False
        """
        new_point = self.points[index]
        old_point = self.points[index - 1]
        return self._test_converge(old_point, new_point)

    def verify_convergence_for_latest_point(self):
        """to test the convergence for the latest point in self.points
        
        Returns:
            bool: True if it converged, otherwise False
        """
        return self.verify_convergence_for_a_point(self.latest_index)

    def _change_trust_radius_step(self, index, multiplier):
        point = self.points[index]
        new_control = point.step_control * multiplier
        point.step_control = max(new_control, self._trust_radius.min)

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
        trust_radius_update_method = TrialOptimizer.trm_update_method[method]
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
        """calculate the secand_condition variable y
        
        Args:
            point (Ts_Treat): the points whose secand_condition you want to calculate for
            point_old (Ts_Treat): the older point you need its information 
        
        Returns:
            numpy.array: the value of the secand_condition variable y
        """
        part1 = point.v_gradient - point_old.v_gradient
        part2 = np.dot(point.v_matrix.T,
                       np.linalg.pinv(point.ts_state.b_matrix).T)
        part3 = np.dot(np.dot(point.ts_state.b_matrix.T,
                              (point.v_matrix - point_old.v_matrix)), point.v_gradient)
        part4 = np.dot((point.ts_state.b_matrix -
                        point_old.ts_state.b_matrix).T, point.ts_state.ic_gradient)
        secant_value = part1 - np.dot(part2, (part3 + part4))
        return secant_value

    @property
    def counter(self):
        """counter of iteration of optimization
        
        Returns:
            int: times of iteration have done
        """
        return self._counter
    
    def _counter_add(self):
        self._counter += 1

    trm_class = {
        "default" : default_trust_radius,
    }

    trm_update_method = {
        "energy" : default_trust_radius._energy_based_trust_radius_method,
        "gradient" : default_trust_radius._gradient_based_trust_radius_method,
    }

    hessian_update_method = {
        'SR1' : hu._sr1_update_method,
        'PSB' : hu._psb_update_method,
        'BFGS' : hu._bfgs_update_method,
        'Bofill' : hu._bofill_update_method,
    }




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
