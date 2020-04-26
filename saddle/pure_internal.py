from abc import ABC

class AbsIC(ABC):

    def __init__(self, atoms, value, ic_type):
        self._atoms = atoms
        self._value = value
        self._type = ic_type

    @property
    def atoms(self):
        return self._atoms

    @property
    def value(self):
        return self._value

    @property
    def type(self):
        return self._type

class Bond(AbsIC):

    def __init__(self, atoms, value):
        super().__init__(atoms, value, ic_type='bond')

class Angle(AbsIC):

    def __init__(self, atoms, value):
        super().__init__(atoms, value, ic_type='angle')

class Dihed(AbsIC):

    def __init__(self, atoms, value):
        super().__init__(atoms, value, ic_type='dihed')
