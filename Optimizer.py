import numpy as np
import copy


class Point(object):
    def __init__(self, coordinates, value = None, first_deriv = None, second_deriv = None):
        self.coordinates = coordinates
        self.value = value
        self.first_deriv = first_deriv
        self.second_deriv = second_deriv

    def __repr__(self): 
        return "point(coordinates: %s, \nvalue: %s, \nfirst_deriv: %s, \nsecond_deriv: %s)"%(self.coordinates, self.value, self.first_deriv,self.second_deriv)



class Procedure(object): # use Procedure object to create more procedure related attribute rather than modify the point object directly
    def __init__(self, point):
        self.point = point #point is a Point object



class Optimizer(object):
    def __init__(self, point, request_object, retrieve_method):
        self.p0 = Procedure(point) 
        self.p1 = None
        self.request_object = request_object
        self.retrieve_method = retrieve_method



    def algorithm(self):
        raise NotImplementedError


class M_S(Optimizer):
    def __init__(self, point, request_object, retrieve_method):
        super(M_S, self).__init__(point, request_object, retrieve_method)


    def algorithm(self):
        self.first_step()
        self.second_step()
        self.third_step()


    def first_step(self):
        self.p0.stepratio = 0.5
        self.p0.G = np.identity(self.len(self.p0.point.coordinates))
        while True:
            steplength = - self.p0.stepratio * np.dot(self.p0.G, self.p0.point.first_deriv)
            new_coordinate = self.p0.point.coordinates + steplength
            new_point = Point(new_coordinate)
            new_point = request_object.retrieve_method(new_point)
            if new_point.value < self.p0.value:
                break
            self.p0.stepratio = self.p0.stepratio * 0.5
        self.p1 = Procedure(new_point) #establish the first step towards optimization
        self.p1.stepratio = self.p0.stepratio


    def second_step(self):
        self.p1.U = -self.p0.stepratio * np.dot(self.p0.G, self.p0.point.first_deriv) - np.dot(self.p0.G, (self.p1.point.first_deriv - self.p0.point.first_deriv))
        self.p1.a_1 = np.dot(np.transpose(self.p1.U), (self.p1.point.first_deriv - self.p0.point.first_deriv))
        self.p1.T = np.dot(np.transpose(self.p1.U), self.p1.U)
        if (self.p1.a_1 < 10**-5 * self.p1.T) or (1.0 / self.p1.a_1 * np.dot(np.transpose(self.p1.U), self.p0.point.first_deriv) > 10**-5):
            self.p1.G = np.identity(self.len(self.p1.point.cooridnates))
            self.p1.stepratio = 0.5
        else:
            self.p1.G = self.p0.G + 1.0 / self.p1.a_1 * np.dot(np.transpose(self.p1.U), self.p1.U)
        steplength = - self.p1.stepratio * (np.dot(self.p1.G, self.p1.point.first_deriv))
        new_coordinate = self.p1.point.coordinates + steplength
        new_point = request_object.retrieve_method(Point(new_coordinate))
        self.new_point = new_point


    def third_step(self):
        self._third_energy_compare()
        self._third_next_step()


        return self.new_point


    def _third_energy_compare(self):
        n = 0
        while (self.new_point.value > self.p1.point.value):
            self.p1.stepratio *= 0.5
            self.second_step()
            n += 1
            if n > 20: break


    def _third_next_step(self):
        n = 0
        while (np.dot(np.transpose(self.new_point.first_deriv), self.new_point.first_deriv) / len(self.new_point.coordinates) > 10 ** -3):
            self.p0 = self.p1
            self.p1 = Procedure(self.new_point)
            self.second_step()
            n += 1
            if n > 20: break        



class DOM(Optimizer):
    def __init__(self, point, request_object, retrieve_method):
        super(DOM, self).__init__(point, request_object, retrieve_method)


    def algorithm(self):
        self.first_step()
        self.second_step()


    def first_step(self):
        self.p0.stepratio = 0.5
        self.p0.G = np.linalg.inv(self.p0.point.second_deriv)
        n = 0
        while True:
            steplength = -self.p0.stepratio * np.dot(self.p0.G, self.p0.point.first_deriv)
            new_coordinate = self.p0.point.coordinates + steplength
            new_point = Point(new_coordinate)
            new_point = self.retrieve_method(self.request_object, new_point)
            n += 1
            if new_point.value < self.p0.point.value:
                break
            self.p0.tepratio *= 0.5
            if n > 20: break
        self.p1 = Procedure(new_point)
        self.p1.stepratio = self.p0.stepratio


    def calculate_new_point(self):
        self.p1.G = np.linalg.inv(self.p1.point.second_deriv)
        steplength = -self.p1.stepratio * np.dot(self.p1.G, self.p1.point.first_deriv) 
        new_coordinate = self.p1.point.coordinates + steplength
        new_point = Point(new_coordinate)
        new_point = self.retrieve_method(self.request_object, new_point)
        print "aaaa"
        self.new_point = new_point


    def second_step(self):
        self.calculate_new_point()
        while self.new_point.value > self.p1.point.value:
            self.p1.stepratio *= 0.5
            self.calculate_new_point()

        if np.dot(self.new_point.first_deriv, self.new_point.first_deriv)/len(self.new_point.coordinates) > 0.001:
            self.p0 = self.p1
            self.p1 = Procedure(self.new_point)
            self.p1.stepratio = self.p0.stepratio
            self.second_step()

        
            
            





        # print "n = %s"%n
        # steplength = -self.p0.stepratio * np.dot(self.p0.G, self.p0.point.first_deriv)
        # new_coordinate = self.p0.point.coordinates + steplength
        # new_point = Point(new_coordinate)
        # new_point = self.retrieve_method(self.request_object, new_point)
        # print new_point.value