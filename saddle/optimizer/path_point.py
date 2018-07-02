import numpy as np
from copy import deepcopy

from saddle.errors import NotSetError


class PathPoint:
    def __init__(self, red_int):
        self._instance = red_int
        self._step = None
        self._stepsize = None
        self._mod_hessian = None

    @property
    def instance(self):
        return self._instance

    @property
    def energy(self):
        return self._instance.energy

    @property
    def x_gradient(self):
        return self._instance.energy_gradient

    @property
    def x_hessian(self):
        return self._instance.energy_hessian

    @property
    def b_matrix(self):
        return self._instance.b_matrix

    @property
    def q_gradient(self):
        return self._instance.q_gradient

    # @property
    # def q_hessian(self):
    #     return self._instance.q_hessian

    @property
    def vspace(self):
        return self._instance.vspace

    @property
    def v_gradient(self):
        return self._instance.v_gradient

    @property
    def v_hessian(self):
        if self._mod_hessian is not None:
            return self._mod_hessian
        return self.raw_hessian

    @v_hessian.setter
    def v_hessian(self, value):
        if self._mod_hessian is not None:
            if self._mod_hessian.shape != value.shape:
                raise ValueError("The shape of input is not valid")
            if not np.allclose(value, value.T):
                raise ValueError("The input Hessian is not hermitian")
        self._mod_hessian = value.copy()

    @property
    def key_ic_number(self):
        return self._instance.key_ic_number

    @property
    def df(self):
        return self._instance.df

    @property
    def raw_hessian(self):
        return self._instance.v_hessian

    @property
    def step(self):
        if self._step is not None:
            return self._step
        raise NotSetError

    @step.setter
    def step(self, value):
        if np.linalg.norm(value) - self.stepsize > 1e-3:
            raise ValueError
        self._step = value.copy()

    @property
    def stepsize(self):
        if self._stepsize is not None:
            return self._stepsize
        raise NotSetError

    @stepsize.setter
    def stepsize(self, value):
        assert value > 0
        self._stepsize = value

    def __repr__(self):
        return f"PathPoint object"

    def run_calculation(self, *_, method):
        self._instance.energy_calculation(method)

    def update_coordinates_with_delta_v(self, step_v):
        # this function will change the coordinates of instance
        self._instance.update_to_new_structure_with_delta_v(step_v)
        # initialize all the private variables
        self._step = None
        self._mod_hessian = None
        self._stepsize = None

    def copy(self):
        return deepcopy(self)

    # TODO: rewrap the lower level function and test
    def fd_hessian(self, coord, *_, eps=0.001, method='g09'):
        if coord >= self.key_ic_number:
            raise ValueError(
                'given coordinates index is not a key internal coordinates')
        # create a perturbation
        unit_vec = np.zeros(self.df)
        unit_vec[coord] = eps
        new_pp = self.copy()
        new_pp.update_coordinates_with_delta_v(unit_vec)
        new_pp.run_calculation(method=method)
        # calculate the finite hessian
        result = self._calculate_finite_diff_h(self, new_pp)
        # assgin result to the column and row
        self._mod_hessian[:, coord] = result
        self._mod_hessian[coord, :] = result

    @staticmethod  # TODO: need test
    def _calculate_finite_diff_h(origin, new_point, eps=0.001):
        # calculate
        d_gv = (new_point.v_gradient - origin.v_gradient) / eps
        d_v = (new_point.vspace - origin.vspace) / eps
        d_b = (new_point.b_matrix - origin.b_matrix) / eps
        part1 = d_gv
        part2 = np.dot(np.dot(origin.b_matrix.T, d_v), origin.v_gradient)
        part3 = np.dot(d_b.T, origin.q_gradient)
        multiply = np.dot(origin.vspace.T, np.linalg.pinv(origin.b_matrix.T))
        result = part1 - np.dot(multiply, (part2 + part3))
        return result
