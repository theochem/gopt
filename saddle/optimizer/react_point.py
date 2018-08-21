import numpy as np
from saddle.optimizer.path_point import PathPoint


class ReactPoint(PathPoint):
    def __init__(self, red_int, dir_vect):
        super().__init__(red_int)
        self._raw_dir_vect = dir_vect

    @property
    def dir_vect(self):
        real_dir_vect = np.dot(
            self.b_matrix,
            np.dot(np.linalg.pinv(self.b_matrix), self._raw_dir_vect))
        return real_dir_vect / np.linalg.norm(real_dir_vect)

    @property
    def vspace(self):
        sub_space = np.outer(self.dir_vect, self.dir_vect)
        return self._instance.vspace - np.dot(sub_space, self._instance.vspace)

    @property
    def v_gradient(self):
        return np.dot(self.vspace.T, self.q_gradient)

    @property
    def v_hessian(self):
        if self._mod_hessian is None:
            return np.dot(np.dot(self.vspace.T, self.q_hessian), self.vspace)
        return self._mod_hessian

    @property
    def sub_q_gradient(self):
        return np.dot(self.vspace, self.v_gradient)

    @property
    def sub_x_gradient(self):
        return np.dot(self.b_matrix.T, self.sub_q_gradient)\

    @v_hessian.setter
    def v_hessian(self, value):
        if self._mod_hessian is not None:
            if self._mod_hessian.shape != value.shape:
                raise ValueError("The shape of input is not valid")
            if not np.allclose(value, value.T):
                raise ValueError("The input Hessian is not hermitian")
            print('Overwrite old mod_hessian')
        self._mod_hessian = value.copy()
