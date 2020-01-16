import numpy as np

from saddle.periodic.periodic import periodic
from saddle.periodic.units import amu


class Procrustes(object):
    def __init__(self, target, *candidates):
        assert len(candidates) > 0
        assert all(np.array_equal(target.numbers, i.numbers) for i in candidates)
        self._target = target
        self._candidates = candidates
        self._target_center = Procrustes._barycenter(target.coordinates, target.numbers)

    @staticmethod
    def _barycenter(coordinates, numbers):
        assert isinstance(coordinates, np.ndarray)
        assert isinstance(numbers, np.ndarray)
        assert numbers.ndim == 1
        assert coordinates.ndim == 2
        fac_ato_ma = np.vectorize(Procrustes._fetch_atomic_mass)
        atom_mass = fac_ato_ma(numbers)
        return np.einsum("i,ij->j", atom_mass, coordinates) / np.sum(atom_mass)

    @staticmethod
    def _fetch_atomic_mass(atomic_num):
        return periodic[atomic_num].mass / amu

    @staticmethod
    def _move_to_center(coordinates, numbers):
        center = Procrustes._barycenter(coordinates, numbers)
        return coordinates - center

    @staticmethod
    def _rotate_coordinates(co1, co2):
        assert isinstance(co1, np.ndarray)
        assert isinstance(co2, np.ndarray)
        assert co1.shape == co2.shape
        if co1.ndim == 1:
            op = np.outer(co1, co2)
        else:
            op = np.dot(co1.T, co2)
        u, _, v = np.linalg.svd(op)
        # return np.dot(np.dot(u, v), co2.T).T
        return np.dot(co2, np.dot(u, v).T)

    def rotate_mols(self):
        adj_tar_coor = Procrustes._move_to_center(
            self._target.coordinates, self._target.numbers
        )
        for i in self._candidates:
            adj_center = Procrustes._move_to_center(i.coordinates, i.numbers)
            adj_can_coor = Procrustes._rotate_coordinates(adj_tar_coor, adj_center)
            yield adj_can_coor + self._target_center
