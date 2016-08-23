from __future__ import print_function, absolute_import
import numpy as np

class optimizer(object):

    def __init__(self, trust_radius, hessian_updater):
        self._points = []
        self._trust_radius = trust_radius
        self._hessian_updater = hessian_updater
