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
        return np.dot(np.dot(self.vspace.T, self.q_hessian), self.vspace)

    @property
    def sub_q_gradient(self):
        return np.dot(self.vspace, self.v_gradient)

    @property
    def sub_x_gradient(self):
        return np.dot(self.b_matrix.T, self.sub_q_gradient)
