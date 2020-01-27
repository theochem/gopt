"""Procrustes module."""
import numpy as np

from saddle.periodic.periodic import periodic
from saddle.periodic.units import amu


class Procrustes(object):
    """Procrustes module for aligning or rotating molecules conformations."""

    def __init__(self, target, *candidates):
        """Initialize Procrustes instance.

        This class is used to align candidate(s) molecules to align with target
        molecule with maximum overlap.

        Parameters
        ----------
        target : np.ndarray(N, 3)
            Coordinates of target molecule
        *candidates : np.ndarray(N, 3)
            One or more candidate molecule(s) to be rotated to be aligned with
            target molecule
        """
        assert len(candidates) > 0
        assert all(np.array_equal(target.numbers, i.numbers) for i in candidates)
        self._target = target
        self._candidates = candidates
        self._target_center = Procrustes._barycenter(target.coordinates, target.numbers)

    @staticmethod
    def _barycenter(coordinates, numbers):
        r"""Compute bary center of given molecule.

        .. math::

            r_{bary} = \frac{\sum_i^n m_i r_i} {\sum_i^n m_i}

        Parameters
        ----------
        coordinates : np.ndarray(N, 3)
            Coordinates of each atomic coordinates
        numbers : np.ndarray(N, )
            Atomic number of each atoms

        Returns
        -------
        np.ndarray(3, )
            Coordinates of barycenter of given molecule
        """
        assert isinstance(coordinates, np.ndarray)
        assert isinstance(numbers, np.ndarray)
        assert numbers.ndim == 1
        assert coordinates.ndim == 2
        fac_ato_ma = np.vectorize(Procrustes._fetch_atomic_mass)
        atom_mass = fac_ato_ma(numbers)
        return np.einsum("i,ij->j", atom_mass, coordinates) / np.sum(atom_mass)

    @staticmethod
    def _fetch_atomic_mass(atomic_num):
        """Compute atomic mass in atomic unit.

        Parameters
        ----------
        atomic_num : int
            Atomic number of certain atom

        Returns
        -------
        float
            Atmoic weight in atomic unit
        """
        return periodic[atomic_num].mass / amu

    @staticmethod
    def _move_to_center(coordinates, numbers):
        """Move molecules bary center to origin.

        Parameters
        ----------
        coordinates : np.ndarray(N, 3)
            Coordinates of given molecule
        numbers : np.ndarray(N, )
            Atomic numbers of given molecule

        Returns
        -------
        np.ndarray(N, 3)
            New translated coordinates with bary center at the origin
        """
        center = Procrustes._barycenter(coordinates, numbers)
        return coordinates - center

    @staticmethod
    def _rotate_coordinates(co1, co2):
        """Rotate co2 coordinates to align with co1 coordinates.

        Parameters
        ----------
        co1 : np.ndarray(N, 3)
            Target coordinates to be aligned with
        co2 : np.ndarray(N, 3)
            Object coordinates to be rotated to align with target coords

        Returns
        -------
        np.ndarray(N, 3)
            Rotated object coordinates with maximum overlap with target coords
        """
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
        """Rotate each candidates molecules to align with target molecule.

        Yields
        ------
        np.ndarray(N, 3)
            Rotated molecule coordinates aligned with target molecule.
        """
        # move target molecule center to origin
        adj_tar_coor = Procrustes._move_to_center(
            self._target.coordinates, self._target.numbers
        )
        for i in self._candidates:
            # move each object molecule center to origin
            adj_center = Procrustes._move_to_center(i.coordinates, i.numbers)
            # align each object molecule with target molecule
            adj_can_coor = Procrustes._rotate_coordinates(adj_tar_coor, adj_center)
            yield adj_can_coor + self._target_center
