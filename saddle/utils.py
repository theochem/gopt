from abc import abstractproperty

import numpy as np
from pathlib import Path
from saddle.periodic.periodic import angstrom, periodic


class Utils():
    def __init__(self, numbers, coordinates):
        self.numbers = numbers.copy()
        self.coordinates = coordinates.copy()

    @classmethod
    def load_file(cls, file_path, encoding='utf-8'):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not isinstance(file_path, Path):
            raise TypeError(f'input file path is not a valid type {file_path}')

        if file_path.suffix == '.xyz':
            nums, coors, _ = Utils._load_xyz(file_path, encoding=encoding)
        elif file_path.suffix in ('.gjf', '.com'):
            nums, coors, _, _ = Utils._load_gauss(file_path, encoding=encoding)
        else:
            raise ValueError(f"Not supported file type: {file_path.suffix}")
        return cls(nums, coors)

    @classmethod
    def save_file(cls, file_path, mole, format='xyz', encoding='utf-8'):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not isinstance(file_path, Path):
            raise TypeError(f'input file path is not a valid type {file_path}')
        if not file_path.suffix:
            file_path = file_path.parent / (file_path.name + f'.{format}')
        assert file_path.suffix == f'.{format}'
        if file_path.suffix == '.xyz':
            cls._save_xyz(file_path, mole, encoding=encoding)
        else:
            raise TypeError(
                f'given file format {file_path.suffix} is not supported by GOpt'
            )

    @staticmethod
    def _load_xyz(file_path, encoding='utf-8'):
        assert isinstance(file_path, Path)
        assert file_path.suffix == '.xyz'
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
    def _save_xyz(file_path, mole, encoding='utf-8'):
        assert isinstance(file_path, Path)
        assert file_path.suffix == '.xyz'
        with file_path.open(encoding=encoding, mode='w') as f:
            f.write(f"{len(mole.numbers)}\n")
            title = getattr(mole, 'title', 'XYZ file Created by GOpt')
            f.write(f'{title}\n')
            for index, atom_n in enumerate(mole.numbers):
                atom_sym = periodic[atom_n].symbol
                coor_x, coor_y, coor_z = mole.coordinates[index] / angstrom
                f.write(
                    f'{atom_sym:>2} {coor_x:15.10f} {coor_y:15.10f} {coor_z:15.10f}\n'
                )

    @staticmethod
    def _load_gauss(file_path, encoding='utf-8'):
        assert isinstance(file_path, Path)
        assert file_path.suffix in ('.com', '.gjf')
        with file_path.open(encoding=encoding) as f:
            numbers = []
            coordinates = []
            line = f.readline()
            flag = False  # flag to indicate reached coordinates lines
            while line:
                contents = line.strip().split()
                if len(contents) == 2:
                    if ((contents[0].isdigit() or contents[0][0] == "-")
                            and contents[1].isdigit()):
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
