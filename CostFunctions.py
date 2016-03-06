import numpy as np


class CostFunctions(object):
    @staticmethod
    def direct_square(origin, target):
        return (origin - target) ** 2


    @staticmethod
    def cos_square(origin, target):
        return (np.cos(origin) - np.cos(target)) ** 2


    @staticmethod
    def dihed_square(origin, target):
        return 2 - 2 * np.cos(origin - target)


    @staticmethod
    def direct_diff(origin, target):
        return 2 * (origin - target)


    @staticmethod
    def cos_diff(origin, target):
        return 2 * (np.cos(origin) - np.cos(target)) * -np.sin(origin)


    @staticmethod
    def dihed_diff(origin, target):
        return 2 * np.sin(origin - target)


    @staticmethod
    def direct_diff_2(origin, target):
        return 2


    @staticmethod
    def cos_diff_2(origin, target):
        return 2 * (np.cos(origin) * np.cos(target) - np.cos(2 * target))


    @staticmethod
    def dihed_diff_2(origin, target):
        return 2 * np.cos(origin - target)
