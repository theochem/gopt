from __future__ import absolute_import

from abc import ABC, abstractproperty

__all__ = ('CoordinateTypes', )


class CoordinateTypes(ABC):
    @abstractproperty
    def value(self):
        pass

    @abstractproperty
    def atoms(self):
        pass

    @abstractproperty
    def info(self):
        pass
