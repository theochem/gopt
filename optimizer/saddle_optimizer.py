import numpy as np
import math
import hessian_update as hu
import os
from horton import angstrom

from trust_radius import default_trust_radius
from copy import deepcopy

# from saddle.saddlepoint import SaddlePoint


class TrialOptimizer(object):
    """Optimizer class to optimize geometry and relative information to saddle point

    Attributes:
        points (list): a list of SaddlePoint instance
        trm_class (dict, class property): trust_radius method chooser, availabel method keyword "default"
    """

    def __init__(self, charge=0, spin=1, title="untitled"):
        self.points = []
        self._trust_radius = None
        # self.parents=[]
        self._counter = 0
        self._charge = charge
        self._spin = spin
        self._title = title
        self.create_log_output()

    def _update_hessian_finite_difference(self, index, key_list, method, perturb=0.001):
        """use finite difference method to update hessian if hessian matrix is 
        not provided or it performed terribly

        Args:
            index (int): index of point in self.points list.
            perturb (float, optional): the scale of perturbation added to each 
        dimention for calculation

        """
        point = self.points[index]
        # h_m = np.zeros((point.ts_state.dof, point.ts_state.dof), float)
        kwargs = {}
        kwargs["spin"] = self._spin
        kwargs["charge"] = self._charge
        kwargs["title"] = self._title
        for i in key_list:
            e_pert = np.zeros(point.ts_state.dof)
            print "finite", point.ts_state.ic
            print "coor", point.ts_state.coordinates / angstrom
            print "gm", point.ts_state.gradient_matrix
            print "ic", point.ts_state.ic
            print "ic gm", point.ts_state.ic_gradient
            print "b * xg", np.dot(np.linalg.pinv(point.ts_state.b_matrix.T), point.ts_state.gradient_matrix)
            e_pert[i] = 1. * perturb
            new_point = point.obtain_new_cc_with_new_delta_v(
                e_pert, method, **kwargs)
            print "finite", new_point.ts_state.ic
            print "coor", new_point.ts_state.coordinates / angstrom
            print "gm", new_point.ts_state.gradient_matrix
            print "ic", new_point.ts_state.ic
            print "ic gm", new_point.ts_state.ic_gradient
            print "b * xg", np.dot(np.linalg.pinv(new_point.ts_state.b_matrix.T), new_point.ts_state.gradient_matrix)
            # new_point = new_ts_state.create_a_saddle_point()
            pt1 = (new_point.v_gradient - point.v_gradient) / perturb
            print "pt1", pt1
            pt2 = np.dot(point.v_matrix.T, np.linalg.pinv(
                point.ts_state.b_matrix.T))
            dv = (new_point.v_matrix - point.v_matrix) / perturb
            pt3 = np.dot(point.ts_state.b_matrix.T,
                         np.dot(dv, point.v_gradient))
            db = (new_point.ts_state.b_matrix -
                  point.ts_state.b_matrix) / perturb
            pt4 = np.dot(db.T, point.ts_state.ic_gradient)
            h_m = pt1 - np.dot(pt2, (pt3 + pt4))
            point.v_hessian[:, i] = h_m
            point.v_hessian[i, :] = h_m
        # point.set_ic_x_hessian()

    def set_trust_radius_method(self, **kwmethod):  # checked
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

    def initialize_trm_for_point_with_index(self, index):  # checked
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

    def verify_new_point_with_point(self, index, new_point):
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
        # print father_point.v_gradient
        norm_new = np.linalg.norm(new_point.ts_state.gradient_matrix)
        norm_old = np.linalg.norm(father_point.ts_state.gradient_matrix)
        # print norm_new, norm_old
        if norm_new > norm_old:
            flag = False
            father_point.step_control *= 0.25
            if father_point.step_control < 0.1 * self._trust_radius.min:
                father_point.step_control = self._trust_radius.min
        return flag

    def verify_new_point_with_latest_point(self, new_point):
        """to test new calculated point is competent to be keep, if not, tweak father point
        and recalculate again.

        Args:
            new_point (Ts_Treat): the point calculated to be test

        Returns:
            bool: True if the point is competent, otherwise False
        """
        return self.verify_new_point_with_point(self.latest_index, new_point)

    @property
    def latest_index(self):
        """return the index of latest point

        Returns:
            int: return the index
        """
        return len(self.points) - 1

    def add_a_point(self, point):  # checked
        """add a point to self.points attribute

        Args:
            point (point instance): a new point instance with updated information
        """
        self.points.append(point)
        self._counter_add()
        if self.latest_index > 0:
            self.write_info(self.latest_index - 1)

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
        if point.v_hessian != None:
            print("exists, quit updating")
        else:
            method = TrialOptimizer.hessian_update_method[method]
            secand_value = self._secant_condition(point, pre_point)
            # function need to be added here
            new_value = method(point, pre_point, secand_value)
            point.v_hessian = new_value
            print("finish updating")
        # make sure Hessian is symmetric
        assert np.allclose(point.v_hessian, point.v_hessian.T)

    def update_hessian_for_latest_point(self, **kwmethod):
        """update hessian for the latest point

        Args:
            **kwmethod: keyword args
                method: the method for updating hessian

        """
        self.update_hessian_for_a_point(self.latest_index, **kwmethod)

    def _test_necessity_for_finite_difference(self, index, omega=1.0, nu=1.0):
        """To test hessian matrix after quasi-Newton method performance
        if good enough, then pass, else, use call finite difference to recalculate more 
        accurate hessian matrix

        Args:
            index (int): the index of structure
            omega (float, optional): the coeffcient of \omega, default value is 1.0
            nu (float, optional): the coeffcient of nu, default value is 1.0

        Returns:
            bool: True if the Hessian performs good, no need to update, otherwise False
        """
        point = self.points[index]
        pre_point = self.points[index - 1]
        need_update = []
        for i in range(point.key_ic):
            condition_1 = False
            norm = np.linalg.norm
            # create a perturbation array
            e_pert = np.zeros(point.ts_state.dof)
            e_pert[i] = 1.
            condition_1 = norm(point.v_gradient[
                              i]) > omega * norm(point.v_gradient) / math.sqrt(point.ts_state.dof)
            condition_2 = norm(np.dot(point.v_hessian, e_pert) - np.dot(
                pre_point.v_hessian, e_pert)) > nu * norm(np.dot(pre_point.v_hessian, e_pert))
            print "c1", condition_1
            print "c2", condition_2
            if condition_1 and condition_2:
                need_update.append(i)
        # return True for no need to update through finite difference,
        # otherwise return False.
        return need_update

    def procruste_process_for_latest_point(self, hessian=False):
        index = self.latest_index
        self.procruste_process_for_a_point(index, hessian)

    def procruste_process_for_a_point(self, index, hessian=False):
        point = self.points[index]
        pre_point = self.points[index - 1]
        overlap = np.dot(point.v_matrix.T, pre_point.v_matrix)
        u, s, v = np.linalg.svd(overlap)
        q = np.dot(u, v)
        point.v_matrix = np.dot(point.v_matrix, q)
        if hessian:
            point.get_v_gradient_hessian()
        else:
            point.get_v_gradient()

    def tweak_hessian_for_a_point(self, index):  # checked
        """tweak the hessian for a point in self.points

        Args:
            index (int): the index of point
        """
        point = self.points[index]
        self._tweak_hessian(point)

    def tweak_hessian_for_latest_point(self):  # checked
        """tweak the hessian for the latest point

        """
        self.tweak_hessian_for_a_point(self.latest_index)

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

    def find_stepsize_for_a_point(self, index, **kwmethod):  # checked
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

    def find_stepsize_for_latest_point(self, **kwmethod):  # checked
        """find the proper stepsize for latest point

        Args:
            index (int): the index of point in self.points
            **kwmethod: keyword args
                method: the stepsize determination method
                    choices: "TRIM" method, "RFO" method

        """
        self.find_stepsize_for_a_point(self.latest_index, **kwmethod)

    def update_to_new_point_for_a_point(self, index, hessian=False, **kwmethod):  # chekced
        """update to a new point depent on the information of present point like
        hessian, trust radius method.

        Args:
            index (int): index of point in the attribute self.points

        Returns:
            Ts_Treat: the new point for further treatment and update
        """
        point = self.points[index]
        method = kwmethod.pop("method")
        title = kwmethod.pop("title","untitled")
        kwargs = {}
        if method == "lf":
            pass
        elif method == "gs":
            kwargs["charge"] = self._charge
            kwargs["spin"] = self._spin
            kwargs["title"] = title
        new_point = point.obtain_new_cc_with_new_delta_v(point.stepsize, method, hessian, **kwargs)
        self.procruste_process_for_a_point(index, hessian)
        return new_point

    def update_to_new_point_for_latest_point(self, hessian=False, **kwmethod):  # checked
        """update to a new point depent on the information of the latest point
        """
        return self.update_to_new_point_for_a_point(self.latest_index, hessian, **kwmethod)

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
        condition_1 = (np.linalg.norm(gm) <= 3.e-4)
        print "condition1", condition_1
        condition_2 = (np.abs(point.ts_state.energy -
                              old_point.ts_state.energy) <= 1e-6)
        delta_q = np.dot(point.v_matrix, point.stepsize)
        delta_x = np.dot(np.linalg.pinv(point.ts_state.b_matrix), delta_q)
        print "condition2", condition_2
        condition_3 = (np.max(np.abs(delta_x)) <= 3e-4)
        print "condition3", condition_3
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
        parameter = kwmethod.pop('parameter', None)
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

    def start_iterate_optimization(self, max_iteration_times=100):
        assert (self.latest_index >= 0)
        assert (self._trust_radius != None)
        for i in range(max_iteration_times):
            self.tweak_hessian_for_latest_point()
            self.find_stepsize_for_latest_point(method="TRIM")
            new_point = self.update_to_new_point_for_latest_point()
            if self.verify_new_point_with_latest_point(new_point) == False:
                new_point = self.update_to_new_point_for_latest_point()
            self.add_a_point(new_point)
            if self.verify_convergence_for_latest_point():
                print "converge!"
                break
            self.update_trust_radius_latest_point(method="gradient")
            self.update_hessian_for_latest_point(method="SR1")

    def create_log_output(self):
        pwd = os.path.dirname(os.path.realpath(__file__))
        file_path = "/../test/gauss/" + self._title + ".log"
        with open(pwd + file_path, "w+") as f:
            f.write("\n ----The log file for optimization---- \n")

    def write_info(self, index):
        pwd = os.path.dirname(os.path.realpath(__file__))
        file_path = "/../test/gauss/" + self._title + ".log"
        point = self.points[index]
        with open(pwd + file_path, "a") as f:
            f.write("\n------------infromation for point {} starts------------".format(index))
            f.write("\natom numbers: \n{}\n".format(point.ts_state.numbers))
            f.write("\ntotal energy: \n{}\n".format(point.ts_state.energy))
            f.write("\ncartesian coordinates: \n{}\n".format(point.ts_state.coordinates))
            f.write("\ninternal coordinates: \n{}\n".format(point.ts_state.ic))
            f.write("\nic transformation matrix: \n{}\n".format(point.ts_state.b_matrix))
            f.write("\ncartesian gradient: \n{}\n".format(point.ts_state.gradient_matrix))
            f.write("\ncartesian hessian: \n{}\n".format(point.ts_state.hessian_matrix))
            f.write("\ninternal gradient: \n{}\n".format(point.ts_state.ic_gradient))
            f.write("\ninternal hessian: \n{}\n".format(point.ts_state.ic_hessian))
            f.write("\nideal steps in x: \n{}\n".format(-np.dot(np.linalg.pinv(point.ts_state.hessian_matrix), point.ts_state.gradient_matrix)))
            f.write("\nideal steps in ic: \n{}\n".format(-np.dot(np.linalg.pinv(point.ts_state.ic_hessian), point.ts_state.ic_gradient)))
            f.write("\nvspace transformation matrix:\n{}\n".format(point.v_matrix))
            f.write("\nvspace gradient:\n{}\n".format(point.v_gradient))
            f.write("\nvspace hessian:\n{}\n".format(point.v_hessian))
            f.write("\noptimization step: \n{}\n".format(point.stepsize))
            f.write("\noptimization step control: \n{}\n".format(point.step_control))
            f.write("------------infromation for point {} ends------------\n".format(index))


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
        "default": default_trust_radius,
    }

    trm_update_method = {
        "energy": default_trust_radius._energy_based_trust_radius_method,
        "gradient": default_trust_radius._gradient_based_trust_radius_method,
    }

    hessian_update_method = {
        'SR1': hu._sr1_update_method,
        'PSB': hu._psb_update_method,
        'BFGS': hu._bfgs_update_method,
        'Bofill': hu._bofill_update_method,
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
