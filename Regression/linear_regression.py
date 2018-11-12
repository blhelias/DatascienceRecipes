# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 23:50:47 2018

@author: brieu

TODO: docstring !
"""

from basic_stats import Stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sklearn.linear_model import LinearRegression


class LinReg:
    """
    learning rate is very important in that case ; if it's too low, algorithm that converge
    and keep increasing exponentially !!
    """
    def __init__(self, method: str = "ols",
                 lr=0.0001,
                 precision=0.01,
                 max_iters=10,
                 previous_step_size=1):
        self.method = method
        self.lr = lr
        self.precision = precision
        self.max_iters = max_iters
        self.previous_step_size = previous_step_size
        self.coefs = None
        self._training_history = []

    def compute_points_error(self,points, b, m):
        """
        y = m * x + b
        m is the slope, b is the intercept
        """
        error = 0
        for i in range(len(points)):
            error += (points[1][i] - (m * points[0][i] + b)) ** 2
        return error / float(len(points))

    def __step_gradient(self, x, y, b_cur, m_cur):
        n = float(len(x))
        gradient_b = 0
        gradient_m = 0
        for i in range(len(x)):
            gradient_b += -(2 / n) * (y[i] - ((m_cur * x[i] + b_cur)))
            gradient_m += -(2 / n) * x[i] * (y[i] - (m_cur * x[i] + b_cur))
        new_b = b_cur - (self.lr * gradient_b)
        new_m = m_cur - (self.lr * gradient_m)
        return [new_b, new_m]

    def __run_step_gradient(self, x, y):
        """
        attention l'initialisation de b et m est importante ! !'
        """
        iters = 0
        #initialisation de b et m
        b =  8
        m = 1
        while iters < self.max_iters:
            b, m = self.__step_gradient(x, y, b, m)
            self._training_history.append([b, m])
            iters += 1
        self.coefs = [b, m]

    def __least_square(self, x, y):
        """
        given training values for x and y,
        find the least-squares values of alpha and beta
        """
        beta = Stats().correlation(x, y) * Stats().standard_deviation(y) / \
            Stats().standard_deviation(x)
        alpha = Stats().mean(y) - beta * Stats().mean(x)
        self.beta = [alpha, beta]
    
    def __L(self, x, y):
        """Maximum Likelihood Estmation
        
        Args:
            x (1D array): data
            y (1D array): target
        
        Raises:
            NotImplementedError: formula is the same as OLS
        """

        raise NotImplementedError

    def fit(self, x: np.array, y: np.array):
        """choose which metod to train lin_reg
        
        Args:
            x (np.array): data
            y (np.array): target
        
        """

        if self.method == "gradient_descent":
            self.__run_step_gradient(x, y)
            
        elif self.method == "ols":
            self.__least_square(x, y)
            
        elif self.method == "mle":
            self.__L(x, y)
    
    def plot_history(self, x, y):
        fig = plt.figure("linear regression")
        res = []
        for element in self._training_history:
            a = plt.scatter(x, y, c="red", s=100, edgecolor="black", alpha=0.5)
            ablines_values_ = [element[1] * i + element[0] for i in x]
            b, = plt.plot(x, ablines_values_, color="blue")
            res.append([a, b])
        img = animation.ArtistAnimation(fig, res, interval=500,
                                        blit=True, repeat_delay=100)
        plt.show()
