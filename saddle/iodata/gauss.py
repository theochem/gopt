import numpy as np

from saddle.periodic import angstrom, periodic

__all__ = ('load_com', 'load_gjf')


def load_com(filename):
    """Load a moleculr geometry from a .com file

    Argument
    --------

    filename: str
        The path to load the .com file

    Return
    ------
    dict, keys includes 'title', 'coordinates', 'numbers'.
    """
    numbers = []
    coordinates = []
    with open(filename) as f:
        line = f.readline()
        while line:
            result = line.strip().split()
            if len(result) == 2 and result[0].isdigit() and result[1].isdigit(
            ):
                charge = int(result[0])
                spin = int(result[1])
            if len(result) == 4:
                element = result[0]
                numbers.append(periodic[element].number)
                x = float(result[1]) * angstrom
                y = float(result[2]) * angstrom
                z = float(result[3]) * angstrom
                coordinates.append([x, y, z])
            line = f.readline()
    return {
        'coordinates': np.array(coordinates),
        'numbers': np.array(numbers),
    }


load_gjf = load_com
