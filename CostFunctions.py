import numpy as np


def direct_square(origin, target):
    return (origin - target) ** 2


def cos_square(origin, target):
    return (np.cos(origin) - np.cos(target)) ** 2


def dihed_square(origin, target):
    return 2 - 2 * np.cos(origin - target)


def direct_diff(origin, target):
    return 2 * (origin - target)


def cos_diff(origin, target):
    return 2 * (np.cos(origin) - np.cos(target)) * -np.sin(origin)


def dihed_diff(origin, target):
    return 2 * np.sin(origin - target)


def direct_diff_2(origin, target):
    return 2


def cos_diff_2(origin, target):
    return 2 * (np.cos(origin) * np.cos(target) - np.cos(2 * target))


def dihed_diff_2(origin, target):
    return 2 * np.cos(origin - target)
