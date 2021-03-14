"""Utility class for file input and output."""
import functools
import inspect
import warnings
from pathlib import Path

import numpy as np

from gopt.periodic.periodic import angstrom, periodic


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


class Z2C:
    """Converting Z matrix to Cartesian coordinates."""

    def __init__(self, center=np.zeros(3)):
        """Initialize with a default origin position.

        Parameters
        ----------
        center : np.ndarray(3), optional
            The origin for the very first atom in the molecule
        """
        self._coords = center.reshape(-1, 3)

    @property
    def natom(self):
        """int: the number of atoms in the system."""
        return len(self._coords) if len(self._coords) != 1 else 0

    @property
    def coords(self):
        """np.ndarray(n, 3): coordinates of the system."""
        return self._coords

    def add_z_entry(self, atoms, values):
        """Add a z matrix entry for an atom.

        The first z entry needs 2 atoms [ind1, ind2] and 1 value [bond]
        second entry needs 3 atoms [ind1, ind2, ind3] and 2 value [bond, angle]
        after, needs 4 atoms [ind1, ind2, ind3, ind4] and 3 values [bond, angle, dihed]
        Note: The newly added atom index need to be the last one
        The bond is the distance between the last two indcies
        The angle is the angle between the bond atoms[-3:]
        The dihed is consist of all four atoms between plane atoms[:3], and atoms [-3:]

        Parameters
        ----------
        atoms : list[int]
            indices of atoms for z matrix coordinates
        values : list[float]
            values of [bond, angle & dihed]
        """
        n_atom = len(atoms)
        n_value = len(values)
        if n_value != n_atom - 1:
            raise ValueError(
                "The #atoms need to be #ic + 1\n"
                f"# of atoms: {n_atom}, # of ics: {n_value}"
            )
        if values[0] <= 0:
            raise ValueError(f"Bond needs to be positive\nGot: {values[0]}")
        # Case 1: the first bond
        if self.natom == 0:
            if n_atom != 2:
                raise ValueError(f"Only 2 atoms need.\nGot: {n_atom}")
            self._add_first(atoms, values)
            return
        # Case 2: the third atom
        if not 0 <= values[1] <= np.pi:  # check angle value
            raise ValueError(f"Angle needs to be in range [0, pi]\ngot: {values[1]}")
        if self.natom == 2:
            if n_atom != 3:
                raise ValueError(f"Only 3 atoms need.\nGot: {n_atom}")
            # assert values[0] > 0, ('Distance needs to be positive')
            self._add_second(atoms, values)
            return
        if self.natom >= 3:
            if n_atom != 4:
                raise ValueError(f"Only 4 atoms need.\nGot: {n_atom}")
            self._add_third(atoms, values)
        else:
            raise ValueError("Unexpected error, send a bug report please.")

    def _add_first(self, atoms, value):
        """Add the very first bond, adding the first two atoms."""
        atom_coor = self.coords[0] + np.array([0.0, 0.0, value[0]])
        self._coords = np.vstack((self.coords, atom_coor))

    def _add_second(self, atoms, values):
        """Add the third atom with only a bond and an angle."""
        atom_coor = np.zeros(3)
        bond, angle = values
        atom_coor[1] = bond * np.sin(angle)
        atom_coor[2] = (-1) ** atoms[1] * bond * np.cos(angle)
        self._coords = np.vstack((self.coords, self.coords[atoms[1]] + atom_coor))

    def _add_third(self, atoms, values):
        """Add any atom with a list of bond, angle, and dihedral."""
        atom_coor = np.zeros(3)
        sphere_tf = self._get_sphere_coor(self.coords[atoms[:3]])
        bond_vector_in_sphere = self._get_sphere_xyz(*values)
        change_in_xyz = sphere_tf.T @ bond_vector_in_sphere
        atom_coor = self.coords[atoms[2]] + change_in_xyz
        self._coords = np.vstack((self.coords, atom_coor))

    @staticmethod
    def _get_sphere_coor(coords):
        """Compute new basis in the last connected points."""
        # R21 is Z
        assert coords.ndim == 2
        r01 = coords[1] - coords[0]
        r21 = coords[1] - coords[2]  # z direction
        r01 /= np.linalg.norm(r01)
        r21 /= np.linalg.norm(r21)
        # align r21 as the z direction
        z = r21
        # r01 r21 may not be orthogonal
        y = np.cross(r01, r21)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        return np.vstack((x, y, z))

    @staticmethod
    def _get_sphere_xyz(bond_v, angle_v, dihed_v):
        """Compute xyz in the spherical coodinates."""
        x = bond_v * np.sin(angle_v) * np.cos(dihed_v)
        y = bond_v * np.sin(angle_v) * np.sin(dihed_v)
        z = bond_v * np.cos(angle_v)
        return np.array([x, y, z])
