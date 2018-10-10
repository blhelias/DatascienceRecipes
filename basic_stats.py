# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:28:38 2018

@author: brieu
"""
import math
class Stats:
    def dot(self, v, w):
        """v_1 * w_1 + ... + v_n * w_n"""
        return sum(v_i * w_i
        for v_i, w_i in zip(v, w))

    def sum_of_squares(self,v):
        """v_1 * v_1 + ... + v_n * v_n"""
        return self.dot(v, v)
    
    def mean(self, x):
        return sum(x) / len(x)

    def de_mean(self, x):
        """translate x by subtracting its mean (so the result has mean 0)"""
        x_bar = self.mean(x)
        return [x_i - x_bar for x_i in x]
    
    def variance(self, x):
        """assumes x has at least two elements"""
        n = len(x)
        deviations = self.de_mean(x)
        return self.sum_of_squares(deviations) / (n - 1)
    
    def standard_deviation(self, x):
        return math.sqrt(self.variance(x))

    def covariance(self, x, y):
        n = len(x)
        return self.dot(self.de_mean(x), self.de_mean(y)) / (n - 1)
    
    def correlation(self, x, y):
        stdev_x = self.standard_deviation(x)
        stdev_y = self.standard_deviation(y)
        if stdev_x > 0 and stdev_y > 0:
            return self.covariance(x, y) / stdev_x / stdev_y
        else:
            return 0
        