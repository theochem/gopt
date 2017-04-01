from __future__ import absolute_import

from abc import ABCMeta, abstractmethod, abstractproperty

__all__ = ['CoordinateTypes']

class CoordinateTypes(object):

    __metaclass__ = ABCMeta

    @abstractproperty
    def value(self):
        pass

    @abstractproperty
    def atoms(self):
        pass

    @abstractproperty
    def info(self):
        pass
