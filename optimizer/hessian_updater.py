from __future__ import absolute_import, print_function, division
import numpy as np



class HessianUpdater(object):

    def update(self, point, new_point, **kwargs):
        method = kwargs.pop("method")
        if kwargs:
            raise TypeError('Unexpected **kwargs: {}'.format(kwargs))
        function = self._update_method_low(method)
        new_hessian = function(point, secant)
        new_point.set_hessian(new_hessian)

    def _update_method_low(self, method_name):
        _hessian_update_method = {
            "SR1": self._SR1 ,
            "PSB": self._PSB ,
            "Bofill": self._Bofill ,
            }
        return _hessian_update_method[method_name]

    def _SR1(self, point, secant):
        part_1 = secant - np.dot(point.p_hessian, point.p_step)
        condition = (np.dot(part_1, point.p_step) ** 2) / (np.linalg.norm(part_1) **2 * np.linalg.norm(point.p_step) **2 )

        if condition <= 1e-18:
            return point.p_hessian.copy() # return a copy rather than a reference of Hessian matrix
        else:
            update_part = np.dot(part_1, part_1.T) / np.dot(part_1, point.p_step)
            return point.p_hessian + update_part

    def _PSB(self, point, secant):
        part_1 = point.p_hessian
        part_2 = np.dot(secant, secant.T) / np.dot(secant.T, point.p_step)
        temp = np.dot(point.p_hessian, point.p_step)
        part_3 = np.dot(temp, temp.T) / np.dot(point.p_step.T, temp)
        return part_1 + part_2 - part_3

    def _Bofill(self, point, secant):
        temp = secant - np.dot(point.p_hessian, point.p_step)
        psi = (np.linalg.norm(np.cross(point.p_step, temp)) ** 2) / (np.linalg.norm(point.p_step)**2 * np.linalg.norm(temp)**2)
        return (1 - psi) * self._SR1(point, secant) + psi * self._PSB(point, secant)

a = HessianUpdater()
func = a._update_method_low("SR1")
print(func)
