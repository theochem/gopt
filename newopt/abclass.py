from __future__ import absolute_import

from abc import ABCMeta, abstractmethod, abstractproperty


class Point(object):

    __metaclass__ = ABCMeta
    # def __init__(self, gradient, hessian, ele_number):
    #     self.gradient = gradient
    #     self.hessian = hessian
    #     self.trust_radius = np.sqrt(ele_number)
    #     self.step = None
    #     self._ele = ele_number

    @abstractproperty
    def gradient(self):
        pass

    @abstractproperty
    def hessian(self):
        pass

    @abstractproperty
    def trust_radius_stride(self):
        pass

    @abstractproperty
    def step(self):
        pass

    @abstractproperty
    def value(self):
        pass

    @abstractmethod
    def set_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_trust_radius_stride(self, *args, **kwargs):
        pass


class TrustRadius(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def readjust(self, *args, **kwargs):
        pass

    @abstractproperty
    def ceiling(self):
        pass

    @abstractproperty
    def floor(self):
        pass

    @abstractmethod
    def starting(self):
        pass
