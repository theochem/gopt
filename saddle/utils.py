"""Utility class for file input and output."""
import functools
import inspect
import warnings
from pathlib import Path

import numpy as np

from saddle.periodic.periodic import angstrom, periodic
from saddle.pure_internal import Bond, Angle, Dihed


class Utils:
    """Utility function class."""

    def __init__(self, numbers, coordinates):
        """Initialize Utility class instance class.

        Parameters
        ----------
        numbers : np.ndarray(N,)
            atomic number of atoms
        coordinates : np.ndarray(N, 3)
            atomic coordinates of atoms
        """
        self.numbers = numbers.copy()
        self.coordinates = coordinates.copy()

    @classmethod
    def load_file(cls, file_path, encoding="utf-8"):
        """Load input file from other format.

        Parameters
        ----------
        file_path : Path or str
            path to designated file
        encoding : str, optional
            encoding method of input file

        Returns
        -------
        Utils
            New instance of Utils with coords and numbers from other input type

        Raises
        ------
        TypeError
            input file_path is not a valid type
        ValueError
            given input file is not supported
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not isinstance(file_path, Path):
            raise TypeError(f"input file path is not a valid type {file_path}")

        if file_path.suffix == ".xyz":
            nums, coors, _ = Utils._load_xyz(file_path, encoding=encoding)
        elif file_path.suffix in (".gjf", ".com"):
            nums, coors, _, _ = Utils._load_gauss(file_path, encoding=encoding)
        else:
            raise ValueError(f"Not supported file type: {file_path.suffix}")
        return cls(nums, coors)

    @classmethod
    def save_file(cls, file_path, mole, format="xyz", encoding="utf-8", mode="w"):
        """Save molecule structure to a given file.

        Parameters
        ----------
        file_path : Path or str
            Path of the stored file, posix with '.xyz'
        mole : Molecule
            Molecule instance with numbers and coordinates
        format : str, optional
            stored file format
        encoding : str, optional
            encoding method of output format
        mode : str, optional
            file I/O mode

        Raises
        ------
        TypeError
            if given file posix is not support
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not isinstance(file_path, Path):
            raise TypeError(f"input file path is not a valid type {file_path}")
        if not file_path.suffix:
            file_path = file_path.parent / (file_path.name + f".{format}")
        assert file_path.suffix == f".{format}"
        if file_path.suffix == ".xyz":
            cls._save_xyz(file_path, mole, encoding=encoding, mode=mode)
        else:
            raise TypeError(
                f"given file format {file_path.suffix} is not supported by GOpt"
            )

    @staticmethod
    def _load_xyz(file_path, encoding="utf-8"):
        assert isinstance(file_path, Path)
        assert file_path.suffix == ".xyz"
        with file_path.open(encoding=encoding) as f:
            size = int(f.readline())
            title = f.readline().strip()
            numbers = np.zeros(size)
            coordinates = np.zeros((size, 3), dtype=float)
            for i in range(size):
                contents = f.readline().split()
                numbers[i] = periodic[contents[0]].number
                coordinates[i, 0] = float(contents[1]) * angstrom
                coordinates[i, 1] = float(contents[2]) * angstrom
                coordinates[i, 2] = float(contents[3]) * angstrom
        return numbers, coordinates, title

    @staticmethod
    def _save_xyz(file_path, mole, encoding="utf-8", mode="w"):
        """Save file to .xyz format.

        Parameters
        ----------
        file_path : Path or str
            Path of the stored file, posix with '.xyz'
        mole : Molecule
            Molecule instance with numbers and coordinates
        encoding : str, optional
            encoding method of output format
        mode : str, optional
            file I/O mode
        """
        assert isinstance(file_path, Path)
        assert file_path.suffix == ".xyz"
        with file_path.open(encoding=encoding, mode=mode) as f:
            f.write(f"{len(mole.numbers)}\n")
            title = getattr(mole, "title", "XYZ file Created by GOpt")
            f.write(f"{title}\n")
            for index, atom_n in enumerate(mole.numbers):
                atom_sym = periodic[atom_n].symbol
                coor_x, coor_y, coor_z = mole.coordinates[index] / angstrom
                f.write(
                    f"{atom_sym:>2} {coor_x:15.10f} {coor_y:15.10f} {coor_z:15.10f}\n"
                )

    @staticmethod
    def _load_gauss(file_path, encoding="utf-8"):
        """Load from gaussian input file.

        Parameters
        ----------
        file_path : Path or str
            Path of the stored file, posix with '.xyz'
        encoding : str, optional
            encoding method of output format

        Returns
        -------
        tuple
            molecular property (atomic numbers, atomic coords, charge, multiplicity)
        """
        assert isinstance(file_path, Path)
        assert file_path.suffix in (".com", ".gjf")
        with file_path.open(encoding=encoding) as f:
            numbers = []
            coordinates = []
            line = f.readline()
            flag = False  # flag to indicate reached coordinates lines
            while line:
                contents = line.strip().split()
                if len(contents) == 2:
                    if (contents[0].isdigit() or contents[0][0] == "-") and contents[
                        1
                    ].isdigit():
                        charge, multi = tuple(map(int, contents))
                        flag = True  # reached coorinates lines
                if len(contents) == 4 and flag:
                    atom_num = contents[0]
                    numbers.append(periodic[atom_num].number)
                    coor = list(map(float, contents[1:]))
                    coordinates.append(coor)
                line = f.readline()
        numbers = np.array(numbers)
        coordinates = np.array(coordinates) * angstrom
        return numbers, coordinates, charge, multi


def deprecated(reason):
    """Generate deprecated warning for old functions or classes.

    This code is originally published on stackoverflow: https://bit.ly/2H4Uvgv
    Appreciation to the original author: Laurent & Zero

    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """
    string_types = (type(b""), type(""))

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Deprecated class {name} ({reason})."
            else:
                fmt1 = "Deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter("always", DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                warnings.simplefilter("default", DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Deprecated class {name}."
        else:
            fmt2 = "Deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))



def get_sphere_coor(coords):
    # R21 is Z
    assert coords.ndim == 2
    r01 = coords[1] - coords[0]
    r21 = coords[1] - coords[2]  # z direction
    r01 /= np.linalg.norm(r01)
    r21 /= np.linalg.norm(r21)
    # align r21 as the z direction
    z = r21
    y = np.cross(r01, r21)
    x = np.cross(y, z)
    return np.vstack((x, y, z))

def get_sphere_xyz(bond_v, angle_v, dihed_v):
    x = bond_v * np.sin(angle_v) * np.cos(dihed_v)
    y = bond_v * np.sin(angle_v) * np.sin(dihed_v)
    z = bond_v * np.cos(angle_v)
    return np.array([x, y, z])


def internal_to_cartesian(internal_sets):
    n_atom = 1
    coords = np.zeros((1, 3), dtype=float)
    # for i, ic in enumerate(internal_sets):
    i = 0
    while i < len(internal_sets):
        # select for the first atom
        if n_atom == 1:
            ic = internal_sets[i]
            atom_coor = coords[0] + np.array([0.0, 0.0, ic.value])
            coords = np.vstack((coords, atom_coor))
            n_atom += 1
            i += 1
        # select for the second atom
        elif n_atom == 2:
            atom_coor = np.zeros(3)
            ics = internal_sets[i : i + 2]
            r_val = ics[0].value
            ind_at = ics[0].atoms[0] if ics[0].atoms[0] != n_atom else ics[0].atoms[1]
            a_val = ics[1].value
            atom_coor[1] = r_val * np.sin(a_val)
            atom_coor[2] = coords[ind_at][2] + (-1) ** ind_at * r_val * np.cos(a_val)
            coords = np.vstack((coords, atom_coor))
            n_atom += 1
            i += 2
        # select for following atoms
        elif n_atom >= 3:
            ics = internal_sets[i : i + 3]
            other_atoms = list(ics[2].atoms[:3])
            sphere_tf = get_sphere_coor(coords[other_atoms])
            bond_vector_in_sphere = get_sphere_xyz(ics[0].value, ics[1].value, ics[2].value)
            change_in_xyz = sphere_tf.T @ bond_vector_in_sphere
            atom_coor = coords[other_atoms[2]] + change_in_xyz
            coords = np.vstack((coords, atom_coor))
            n_atom += 1
            i += 3

    return coords
