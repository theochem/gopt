import numpy as np


class FCHKFile(object):
    """Reader for Formatted checkpoint files

       After initialization, the data from the file is available in the fields
       dictionary. Also the following attributes are read from the file: title,
       command, lot (level of theory) and basis.
    """

    def __init__(self, filename, ignore_errors=False, field_labels=None):
        """
           Arguments:
            | ``filename``  --  The formatted checkpoint file

           Optional arguments:
            | ``ignore_errors``  --  Try to read incorrectly formatted files
                                     without raising exceptions [default=False]
            | ``field_labels``  --  When given, only these fields are read from
                                    the formatted checkpoint file. (This can
                                    save a lot of time.)
        """
        self.filename = filename
        self.ignore_errors = ignore_errors
        try:
            if field_labels is not None:
                field_labels = set(field_labels)
                field_labels.add("Atomic numbers")
                field_labels.add("Current cartesian coordinates")
            self._read(filename, field_labels)
        except FileFormatError:
            if ignore_errors:
                pass
            else:
                raise
        self._analyze()

    def _read(self, filename, field_labels=None):
        """Read all the requested fields

           Arguments:
            | ``filename``  --  the filename of the FCHK file
            | ``field_labels``  --  when given, only these fields are read
        """

        # if fields is None, all fields are read
        def read_field(f):
            """Read a single field"""
            datatype = None
            while datatype is None:
                # find a sane header line
                line = f.readline()
                if line == "":
                    return False

                label = line[:43].strip()
                if field_labels is not None:
                    if len(field_labels) == 0:
                        return False
                    elif label not in field_labels:
                        return True
                    else:
                        field_labels.discard(label)
                line = line[43:]
                words = line.split()
                if len(words) == 0:
                    return True

                if words[0] == 'I':
                    datatype = int
                    unreadable = 0
                elif words[0] == 'R':
                    datatype = float
                    unreadable = np.nan

            if len(words) == 2:
                try:
                    value = datatype(words[1])
                except ValueError:
                    return True
            elif len(words) == 3:
                if words[1] != "N=":
                    raise FileFormatError(
                        "Unexpected line in formatted checkpoint file %s\n%s" %
                        (filename, line[:-1]))
                length = int(words[2])
                value = np.zeros(length, datatype)
                counter = 0
                try:
                    while counter < length:
                        line = f.readline()
                        if line == "":
                            raise FileFormatError(
                                "Unexpected end of formatted checkpoint file %s"
                                % filename)
                        for word in line.split():
                            try:
                                value[counter] = datatype(word)
                            except (ValueError, OverflowError), e:
                                print 'WARNING: could not interpret word while reading %s: %s' % (
                                    word, self.filename)
                                if self.ignore_errors:
                                    value[counter] = unreadable
                                else:
                                    raise
                            counter += 1
                except ValueError:
                    return True
            else:
                raise FileFormatError(
                    "Unexpected line in formatted checkpoint file %s\n%s" %
                    (filename, line[:-1]))

            self.fields[label] = value
            return True

        self.fields = {}
        f = file(filename, 'r')
        self.title = f.readline()[:-1].strip()
        words = f.readline().split()
        if len(words) == 3:
            self.command, self.lot, self.basis = words
        elif len(words) == 2:
            self.command, self.lot = words
        else:
            raise FileFormatError(
                'The second line of the FCHK file should contain two or three words.'
            )

        while read_field(f):
            pass

        f.close()

    def _analyze(self):
        """Convert a few elementary fields into a molecule object"""
        if ("Atomic numbers" in self.fields) and (
                "Current cartesian coordinates" in self.fields):
            self.molecule = Molecule(
                self.fields["Atomic numbers"],
                np.reshape(self.fields["Current cartesian coordinates"],
                           (-1, 3)),
                self.title, )

    def get_coordinates(self):
        """Return cartesian coordinates in format check file"""
        return self.fields.get("Current cartesian coordinates")

    def get_hessian(self):
        """Return the hessian"""
        force_const = self.fields.get("Cartesian Force Constants")
        if force_const is None:
            return None
        N = len(self.molecule.numbers)
        result = np.zeros((3 * N, 3 * N), float)
        counter = 0
        for row in xrange(3 * N):
            result[row, :row + 1] = force_const[counter:counter + row + 1]
            result[:row + 1, row] = force_const[counter:counter + row + 1]
            counter += row + 1
        return result

    def get_gradient(self):
        return self.fields.get("Cartesian Gradient")

    def get_energy(self):
        return self.fields.get("Total Energy")


class Molecule(object):
    def __init__(self, numbers, coordinates, title):
        self.numbers = numbers
        self.coordinates = coordinates
        self.title = title


class FileFormatError(Exception):
    pass


if __name__ == '__main__':
    a = FCHKFile("./test/water_0.fchk")
    hessian = a.get_hessian()
    gradient = a.get_gradient()
    energy = a.get_energy()
    # print hessian
    print gradient
    print energy
