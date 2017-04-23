from __future__ import absolute_import

from abc import ABCMeta, abstractmethod, abstractproperty


class Point(object):

    __metaclass__ = ABCMeta

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

    @abstractmethod
    def set_hessian(self, *args, **kwargs):
        pass

    @abstractmethod
    def update_point(self, *args, **kwargs):
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


class HessianModifier(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def modify_hessian(self, *args, **kwargs):
        pass


class StepScaler(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def calculate_step(self, *args, **kwargs):
        pass
