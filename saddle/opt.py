"""Transform optimization process module."""
import numpy as np

from saddle.math_lib import pse_inv, ridders_solver

__all__ = ("Point", "GeoOptimizer")


class Point(object):
    """Point class for holding optimiztaion property."""

    def __init__(self, gradient, hessian, ele_number):
        """Initialize Point class.

        Parameters
        ----------
        gradient : np.ndarray(N,)
            Gradient of a point
        hessian : np.ndarray(N, N)
            Hessian of a point
        ele_number : int
            Number of electron in the molecule
        """
        self.gradient = gradient
        self.hessian = hessian
        self.trust_radius = np.sqrt(ele_number)
        self.step = None
        self._ele = ele_number

    @property
    def ele(self):
        """int: number of electron in the system."""
        return self._ele


class GeoOptimizer(object):
    """Coordinates Transformation optimization class."""

    def __init__(self):
        """Initialize Geo optimization class."""
        self.points = []

    def __getitem__(self, index):
        """Add slicing functionality to class."""
        return self.points[index]

    def converge(self, index):
        """Check given index point converged or not.

        Parameters
        ----------
        index : int
            The index of the point in the points list

        Returns
        -------
        bool
            True if it converged, otherwise False
        """
        point = self.points[index]
        return max(np.abs(point.gradient)) <= 1e-7

    @property
    def newest(self):
        """int: the length of all points."""
        return len(self.points) - 1

    def newton_step(self, index):
        """Compute newtom-raphson step for certain point index.

        Parameters
        ----------
        index : int
            index of point

        Returns
        -------
        np.ndarray
            newton step to be taken in the next iteration
        """
        point = self.points[index]
        return -np.dot(pse_inv(point.hessian), point.gradient)

    def add_new(self, point):
        """Add a new point to the Point class.

        Parameters
        ----------
        point : Point
            new optimization point to be added to the list
        """
        self.points.append(point)

    def tweak_hessian(self, index, negative=0, threshold=0.05):
        """Tweak eigenvalues of hessian to be positive-semi-definite.

        Parameters
        ----------
        index : int
            index of opt point
        negative : int, optional
            number of negative eiganvalues
        threshold : float, optional
            update hessian value if the eigenvalue is too small
        """
        point = self.points[index]
        w, v = np.linalg.eigh(point.hessian)
        negative_slice = w[:negative]
        positive_slice = w[negative:]
        negative_slice[negative_slice > -threshold] = -threshold
        positive_slice[positive_slice < threshold] = threshold
        new_hessian = np.dot(v, np.dot(np.diag(w), v.T))
        point.hessian = new_hessian

    def trust_radius_step(self, index, negative=0):
        """Compute trust radius step for optimization.

        Parameters
        ----------
        index : int
            index of opt point
        negative : int, optional
            number of negative eigenvalues

        Returns
        -------
        np.ndarray
            new step controlled by trust radius
        """
        point = self.points[index]
        c_step = self.newton_step(index)
        if np.linalg.norm(c_step) <= point.trust_radius:
            point.step = c_step
            return c_step
        w, v = np.linalg.eigh(point.hessian)
        # incase different step calculated from tests
        max_w = round(max(w), 7)

        def func_step(value):
            """Compute proper update step."""
            x = w.copy()
            x[:negative] = x[:negative] - value
            x[negative:] = x[negative:] + value
            new_hessian_inv = np.dot(v, np.dot(np.diag(1.0 / x), v.T))
            return -np.dot(new_hessian_inv, point.gradient)

        def func_value(value):
            """Compute function value difference."""
            step = func_step(value)
            return np.linalg.norm(step) - point.trust_radius

        while func_value(max_w) >= 0:
            max_w *= 2
        result = ridders_solver(func_value, 0, max_w)
        result = round(result, 7)  # incase different test result
        # print ("result", result)
        step = func_step(result)
        point.step = step
        return step

    def update_trust_radius(self, index):
        """Update trust radius for given index point."""
        point = self.points[index]
        pre_point = self.points[index - 1]
        if np.linalg.norm(point.gradient) > np.linalg.norm(pre_point.gradient):
            point.trust_radius = pre_point.trust_radius * 0.25
            return
        g_predict = pre_point.gradient + np.dot(pre_point.hessian, pre_point.step)
        if np.linalg.norm(point.gradient) - np.linalg.norm(pre_point.gradient) == 0:
            ratio = 3.0
            # if the gradient change is 0, then use the set_trust_radius
        else:
            ratio = np.linalg.norm(g_predict) - np.linalg.norm(pre_point.gradient) / (
                np.linalg.norm(point.gradient) - np.linalg.norm(pre_point.gradient)
            )
        if 0.8 <= ratio <= 1.25:
            point.trust_radius = pre_point.trust_radius * 2.0
        elif 0.2 <= ratio <= 6:
            point.trust_radius = pre_point.trust_radius
        else:
            point.trust_radius = pre_point.trust_radius * 0.5
        point.trust_radius = min(
            max(point.trust_radius, 0.1 * np.sqrt(point.ele)), 2.0 * np.sqrt(point.ele)
        )
