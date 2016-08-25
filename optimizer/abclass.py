from __future__ import absolute_import
from abc import ABCMeta, abstractmethod, abstractproperty

class Points(object):

    __metaclass__ = ABCMeta

    @abstractproperty
    def p_coordinates(self):
        pass

    @abstractproperty
    def p_energy(self):
        pass

    @abstractproperty
    def p_gradient(self):
        pass

    @abstractproperty
    def p_hessian(self):
        pass

    @abstractproperty
    def p_trust_radius(self):
        pass

    @abstractproperty
    def p_step(self):
        pass

    @abstractmethod
    def set_hessian():
        pass

    @abstractmethod
    def set_trust_radius():
        pass


class TrustRadius(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def update(self):
        pass
