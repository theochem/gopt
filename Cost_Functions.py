import numpy as np
import molmod as mm


class Cost_Function(object):

    @staticmethod
    def bond_cost(origin, target):
        value = (origin - target) ** 2
        deriv = 2 * origin - 2 * target


    @staticmethod
    def angle_cost(origin, target):
        value = (np.cos(origin) - cos(target)) ** 2
        deriv = (2 * np.cos(origin) - 2 * np.cos(target)) * -np.sin(origin)


    @staticmethod
    def dihed_cost(origin, target, coordinates):
        angle1 = mm.bend_angle(np.array([coordinates[0], coordinates[1], coordinates[2]]))
        angle2 = mm.bend_angle(np.array([coordinates[1], coordinates[2], coordinates[3]]))
        angle_wight = np.sin(angle1) ** 2 * np.sin(angle2) ** 2
        value = angle_wight * ((np.cos(origin) - np.cos(target)) ** 2 + (np.sin(origin) - np.sin(target)) ** 2)
        deriv = angle_wight * ((2 * np.cos(origin) - 2 * np.cos(target)) * -np.sin(origin)) + ((2 * np.sin(origin) - 2 * np.sin(target)) * np.cos(target))


    @staticmethod
    def dihed_new_cost(origin, target):
        value = (origin - target) ** 2
        deriv = 2 * origin - 2 * target