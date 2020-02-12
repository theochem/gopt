"""Reaction Path point instance used in optimization process."""

from copy import deepcopy

import numpy as np

from saddle.errors import NotSetError
from saddle.math_lib import pse_inv


class PathPoint:
    """PathPoint class for optimization."""

    def __init__(self, red_int):
        """Initialize a PathPoint for optimization process.

        Parameters
        ----------
        red_int : ReducedInternal
            an structure reducedinternal instance
        """
        self._instance = red_int
        self._step = None
        self._stepsize = None
        self._mod_hessian = None
        self._step_hessian = None

    @property
    def instance(self):
        """ReducedInternal: The reduced internal instance correspound to this point."""
        return self._instance

    @property
    def energy(self):
        """float: the energy of reduced internal."""
        return self._instance.energy

    @property
    def x_gradient(self):
        """np.ndarray(3N,): the cartesian gradient of reduced internal."""
        return self._instance.energy_gradient

    @property
    def x_hessian(self):
        """np.ndarray(3N, 3N): the cartesian Hessian of reduced internal."""
        return self._instance.energy_hessian

    @property
    def b_matrix(self):
        """np.ndarray(3N, n): the cartesian to internal transform gradient."""
        return self._instance.b_matrix

    @property
    def q_gradient(self):
        """np.ndarray(n,): the internal gradient of reduced internal."""
        return self._instance.q_gradient

    @property
    def q_hessian(self):
        """np.ndarray(n, n): the internal hessian of reduced internal."""
        return self._instance.q_hessian

    @property
    def vspace(self):
        """np.ndarray(n, 3N-6): the transform matrix from reduced to internal."""
        return self._instance.vspace

    @property
    def v_gradient(self):
        """np.ndarray(3N-6): the vspace gradient of reduced internal."""
        return self._instance.v_gradient

    @property
    def v_hessian(self):
        """np.ndarray(3N-6, 3N-6): the vspace Hessian of reduced internal."""
        if self._mod_hessian is not None:
            return self._mod_hessian
        return self.raw_hessian

    @property
    def step_hessian(self):
        """np.ndarray(3N, 3N): updated and modified Hessian matrix for step calculation."""
        if self._step_hessian is None:
            raise NotSetError("Step hessian is not set yet")
        return self._step_hessian

    @step_hessian.setter
    def step_hessian(self, value):
        """Set a new value to cartesian Hessian matrix."""
        assert value.shape == self.v_hessian.shape
        self._step_hessian = value

    @v_hessian.setter
    def v_hessian(self, value):
        """Set a new to vspace hessian."""
        if self._mod_hessian is not None:
            if self._mod_hessian.shape != value.shape:
                raise ValueError("The shape of input is not valid")
            if not np.allclose(value, value.T):
                raise ValueError("The input Hessian is not hermitian")
            print("Overwrite old mod_hessian")
        self._mod_hessian = value.copy()

    @property
    def key_ic_number(self):
        """int: number of key internal coordinates."""
        return self._instance.key_ic_number

    @property
    def df(self):
        """int: degree of freedom of given molecule, normally 3N - 6."""
        return self._instance.df

    @property
    def raw_hessian(self):
        """np.ndarray(3N, 3N): original cartesian hessian before modification."""
        return self._instance.v_hessian

    @property
    def step(self):
        """np.ndarray(3N-6): new optimization step in v space."""
        if self._step is not None:
            return self._step
        raise NotSetError

    @step.setter
    def step(self, value):
        """Set new step to instance."""
        if np.linalg.norm(value) - self.stepsize > 1e-3:
            raise ValueError
        self._step = value.copy()

    @property
    def stepsize(self):
        """float: up-bound of optimization step."""
        if self._stepsize is not None:
            return self._stepsize
        raise NotSetError

    @stepsize.setter
    def stepsize(self, value):
        """Set a new stepsize for point."""
        assert value > 0
        self._stepsize = value

    def __repr__(self):
        """Show string representation of PathPoint."""
        return f"PathPoint object"

    def run_calculation(self, *_, method):
        """Run calculation for PathPoint instance.

        Parameters
        ----------
        method : str
            name of outer quantum chemistry software to run calculation.
        """
        self._instance.energy_calculation(method)

    def update_coordinates_with_delta_v(self, step_v):
        """Update the struture from the change in Vspace.

        Parameters
        ----------
        step_v : np.ndarray(3N-6)
            structure update changes in vspace
        """
        # this function will change the coordinates of instance
        self._instance.update_to_new_structure_with_delta_v(step_v)
        # initialize all the private variables
        self._step = None
        self._mod_hessian = None
        self._stepsize = None

    def copy(self):
        """Make a deepcopy of the instance itself."""
        return deepcopy(self)

    # TODO: rewrap the lower level function and test
    def fd_hessian(self, coord, *_, eps=0.001, method="g09"):
        """Run finite difference on hessian update.

        Parameters
        ----------
        coord : np.ndarray(N, 3)
            coordinates of molecule
        eps : float, optional
            finite step for finite difference calculation
        method : str, optional
            name of outer quantum chemistry software for calculation

        Raises
        ------
        ValueError
            if the key ic number is not a valid value
        """
        if coord >= self.key_ic_number:
            raise ValueError(
                "given coordinates index is not a key internal coordinates"
            )
        # create a perturbation
        unit_vec = np.zeros(self.df)
        unit_vec[coord] = eps
        new_pp = self.copy()
        new_pp.update_coordinates_with_delta_v(unit_vec)
        new_pp.run_calculation(method=method)
        # align vspace in finite diff
        new_pp.instance.align_vspace(self.instance)
        # calculate the finite hessian
        result = self._calculate_finite_diff_h(self, new_pp, eps=eps)
        # assgin result to the column and row
        self._mod_hessian[:, coord] = result
        self._mod_hessian[coord, :] = result

    @staticmethod  # TODO: need test
    def _calculate_finite_diff_h(origin, new_point, eps):
        # calculate
        d_gv = (new_point.v_gradient - origin.v_gradient) / eps
        d_v = (new_point.vspace - origin.vspace) / eps
        d_b = (new_point.b_matrix - origin.b_matrix) / eps
        part1 = d_gv
        part2 = np.dot(np.dot(origin.b_matrix.T, d_v), origin.v_gradient)
        part3 = np.dot(d_b.T, origin.q_gradient)
        multiply = np.dot(origin.vspace.T, pse_inv(origin.b_matrix.T))
        result = part1 - np.dot(multiply, (part2 + part3))
        return result
