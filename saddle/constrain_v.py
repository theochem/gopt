import warnings
import numpy as np

from saddle.errors import ICNumberError, NotSetError
from saddle.internal import Internal
from saddle.math_lib import diagonalize, pse_inv
from saddle.math_lib import procrustes


class NewVspace(Internal):
    def __init__(self,
                 coordinates: 'np.ndarray',
                 numbers: 'np.ndarray',
                 charge: int,
                 multi: int,
                 title: str = "") -> None:
        super(NewVspace, self).__init__(coordinates, numbers, charge, multi,
                                        title)
        self._n_freeze_ic = 0
        self._n_key_ic = 0
        self._n_dof = len(self.numbers) - 6
        self._freeze_space = None
        self._key_space = None
        self._non_space = None
        self._vspace = None

    @property
    def vspace(self):
        if (self._freeze_space is None or self._key_space is None
                or self._non_space is None):
            self._generate_freeze_space()
            self._generate_key_space()
            self._generate_non_space()
        return np.hstack((self._freeze_space, self._key_space,
                          self._non_space))

    @property
    def vspace_gradient(self):
        return ...

    @property
    def vspace_hessian(self):
        return ...

    @property
    def n_freeze(self):
        return self._n_freeze_ic

    @property
    def n_key(self):
        return self._n_key_ic

    @property
    def n_nonkey(self):
        return self.df - self.n_key - self.n_freeze

    def select_freeze_ic(self, *indices: int) -> None:
        if any(np.array(indices) < 0):
            raise IndexError('Non negative index allowed')
        n_freeze_ic = 0
        indices = np.unique(np.sort(np.array(indices)))
        if max(indices) >= len(self.ic):
            raise IndexError('Given index is out of IC range')

        if self.n_key != 0:
            print('Key ic info will be erased')
            self._n_key_ic = 0

        for index in indices:
            self.swap_internal_coordinates(n_freeze_ic, index)
            n_freeze_ic += 1
        self._n_freeze_ic = n_freeze_ic
        self._reset_v_space()

    def select_key_ic(self, *indices: int) -> None:
        if any(np.array(indices) < 0):
            raise IndexError('Non negative index allowed')
        n_key_ic = 0
        indices = np.unique(np.sort(np.array(indices)))
        if max(indices) >= len(self.ic):
            raise IndexError('Given index is out of IC range')
        if min(indices) < self.n_freeze:
            raise IndexError('Can\'t set frozen ic to key ic')
        for index in indices:
            self.swap_internal_coordinates(n_key_ic + self.n_freeze, index)
            n_key_ic += 1
        self._n_key_ic = n_key_ic
        self._reset_v_space()

    def set_vspace(self, new_vspace):
        if new_vspace.shape != (len(self.ic), self.df):
            raise ValueError('Given new vspace is not in the right shape')
        n_f_k = self.n_freeze + self.n_key
        self._freeze_space = new_vspace[:, :self.n_freeze]
        self._key_space = new_vspace[:, self.n_freeze:n_f_k]
        self._non_space = new_vspace[:, n_f_k:]

    def align_vspace_matrix(self, target, special=False):
        if not isinstance(target, np.ndarray):
            raise TypeError('Input matrix is not a legit numpy array')
        if target.shape != (len(self.ic), self.df):
            raise ValueError(
                f'Input array doesn\'t have a correct shape {target.shape}')
        if special is False:
            new_v = procrustes(self.vspace, target)
            self.set_vspace(new_v)
        else:
            n_f_k = self.n_freeze + self.n_key
            n_f = self.n_freeze
            new_v_freeze = procrustes(self.vspace[:, :n_f], target[:, :n_f])
            new_v_key = procrustes(self.vspace[:, n_f:n_f_k],
                                   target[:, n_f:n_f_k])
            new_v_non = procrustes(self.vspace[:, n_f_k:], target[:, n_f_k:])
            new_v = np.hstack((new_v_freeze, new_v_key, new_v_non))
        self.set_vspace(new_v)

    def align_vspace(self, target, ic_check=False):
        # could add check for same ic selection
        if not isinstance(target, NewVspace):
            raise TypeError('Input molecule is not a valid type')
        if ic_check:
            if target.n_freeze != self.n_freeze or target.n_key != self.n_key:
                raise ValueError(
                    'Number of special cooridnates does not match')
        self.align_vspace_matrix(target.vspace)

    def _reduced_unit_vectors(self, *start_ends) -> 'np.ndarray':  # tested
        """Generate unit vectors where every position is 0 except the
        key_ic_number position is 1

        Returns
        -------
        unit_mtx : np.ndarray(K, J), J is the number of key_ic_number
            Unit vectors with 0 everywhere except that key_ic_number position
            is 0
        """
        if len(start_ends) == 1 and start_ends[0] >= 0:
            start = 0
            end = start_ends[0]
        elif len(start_ends) == 2 and start_ends[1] >= start_ends[0]:
            start, end = start_ends
        else:
            raise ValueError(f'takes 1 or 2 arguments. {start_ends} is given')
        num_col = end - start
        unit_mtx = np.zeros((len(self.ic), num_col))
        unit_mtx[start:end, :] = np.eye(num_col)
        return unit_mtx

    def _reduced_perturbation(self, *start_ends) -> 'np.ndarray':  # tested
        """Calculate the realizable purterbation in internal cooridnates from
        the unit vectors

        Returns
        -------
        purterbs : np.ndarray(K, J)
            Realizable purterbation in internal cooridnates from those unit
            vectors
        """
        unit_mtx = self._reduced_unit_vectors(*start_ends)
        tsfm = np.dot(self.b_matrix, pse_inv(self.b_matrix))
        return np.dot(tsfm, unit_mtx)

    def _generate_freeze_space(self, threshold=1e-6):
        b_mtx = self._reduced_perturbation(self.n_freeze)
        w, v = diagonalize(b_mtx)
        self._freeze_space = v[:, np.abs(w) > threshold]

    def _generate_key_space(self, threshold=1e-6):
        b_mtx = self._reduced_perturbation(self.n_freeze,
                                           self.n_key + self.n_freeze)
        # project out freezed space
        proj_f = np.dot(self._freeze_space, self._freeze_space.T)
        left_b_mtx = b_mtx - np.dot(proj_f, b_mtx)
        w, v = diagonalize(left_b_mtx)
        self._key_space = v[:, np.abs(w) > threshold]

    def _generate_non_space(self, threshold=1e-6):
        a_mtx = self._svd_of_b_matrix()
        red_space = np.hstack((self._freeze_space, self._key_space))
        prj_rd = np.dot(red_space, red_space.T)
        non_red_v = a_mtx - np.dot(prj_rd, a_mtx)
        w, v = diagonalize(non_red_v)
        self._non_space = v[:, abs(w) > threshold]

    def _svd_of_b_matrix(self, threshold=1e-6) -> 'np.ndarray':  # tested
        """Select 3N - 6 non-singular vectors from b_matrix through SVD
        (eigenvalue) decomposition

        Returns:
        u : np.ndarray(K, 3N - 6)
            3N - 6 non-singular vectors from SVD
        """
        # use eigh rather than svd, more likely to get unique result
        b_space = np.dot(self.b_matrix, self.b_matrix.T)
        values, vectors = np.linalg.eigh(b_space)
        # b_matrix shape is n * 3N
        # b_space is n * n
        basis = vectors[:, np.abs(values) > threshold]
        # for nonlinear molecules
        # TODO: need change for nonlinear molecules
        assert len(basis[0]) == self.df
        return basis

    def _reset_v_space(self):
        self._freeze_space = None
        self._key_space = None
        self._non_space = None

    def _add_cc_to_ic_gradient(self, deriv, atoms):
        super()._add_cc_to_ic_gradient(deriv, atoms)
        self._reset_v_space()

    def _clear_ic_info(self):
        super()._clear_ic_info()
        self._reset_v_space()
