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

from sklearn.linear_model import LinearRegression


class LinReg:
    """
    learning rate is very important in that case ; if it's too low, algorithm that converge
    and keep increasing exponentially !!
    """
    def __init__(self,
                 gradient_descent=False,
                 least_square=False,
                 mle=False,
                 lr=0.0001,
                 precision=0.01,
                 max_iters=10000,
                 previous_step_size=1):
        self.gradient_descent = gradient_descent
        self.least_square = least_square
        self.mle = mle
        self.lr = lr
        self.precision = precision
        self.max_iters = max_iters
        self.previous_step_size = previous_step_size
        self.coefs = None

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

        if self.gradient_descent:
            self.__run_step_gradient(x, y)
            
        elif self.least_square:
            self.__least_square(x, y)
            
        elif self.mle:
            self.__L(x, y)

if __name__ == "__main__":
    def compute_results(x,y,b,m,title,color):
        """
        param:
            colonne X
            colonne Y
            coeff M (pente)
            B (intercept)
            titre du graph
            couleur de la droite de regression
        """


        print(" {} : b = {} ; m = {}".format(title,b,m))
        plt.figure(title)
        plt.scatter(x,y,c=color)
        ablines_values = [m * i + b for i in x]
        plt.plot(x,ablines_values)
        plt.title(title)

        plt.show()
    
    """
    1 - chargement du dataset
    """
    # logging.warning('[INFO] chargement du dataset')
    data = pd.read_table("../data/data.txt",sep="\t",header=None)
    x= np.array(data[0])
    y= np.array(data[1])

    """
    2 - regression descente de gradient
    """
    # logging.warning('[INFO] entrainement regression')
    lin_reg_gradient_descent = LinReg(gradient_descent=True)

    # train liner regression model gradient descent
    b,m = lin_reg_gradient_descent.fit(x,y)
    title_dg = "gradient descent"
    compute_results(x, y, b, m, title_dg, "blue")

    """
    3 - regression moindre carrés
    """

    lin_reg_least_square = LinReg(least_square=True)
    #train linear regression model least_square
    b,m = lin_reg_least_square.fit(x,y)
    title_ls = "least square"
    compute_results(x,y,b,m,title_ls,"red")

    """
    4 - regression Scikit-Learn
    """
    regr = LinearRegression(fit_intercept=True)    
    # Train the model using the training sets
    x=x.reshape((100,1))
    y=y.reshape((100,1))
    
    regr.fit(x, y)

    b = regr.intercept_[0]
    m = regr.coef_[0][0]
    title = "scikit learn"
    compute_results(x,y,b,m,title,"green")
    plt.show()