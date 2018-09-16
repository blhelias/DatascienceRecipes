# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 23:50:47 2018

@author: brieu
"""


"""
###########################################
#this algorithm applies for diension 1
#( From wikipedia )
###########################################

X_0 = 5 # The algorithm starts at x=5
GAMMA = 0.01 # step size multiplier (learning rate)
precision = 0.001
previous_step_size = 1
max_iters = 10000 # maximum number of iterations
iters = 0 #iteration counter

df = lambda x: 4 * x**3 - 33 * x**2 + 82 * x - 61

while (previous_step_size > precision) & (iters < max_iters):
    prev_x = X_0
    X_0 -= GAMMA * df(prev_x)
    previous_step_size = abs(X_0 - prev_x)
    iters+=1

print("The local minimum occurs at", X_0,"for ",iters," iterations")
#The output for the above will be: ('The local minimum occurs at', 2.2499646074278457)

############################################
this algorithm applies for diension 2 F(x,y)
"""

from basic_stats import Stats


class LinearRegressionBrieuc:
    """
    learning rate is very important in that case ; if it's too low, algorithm that converge
    and keep increasing exponentially !!
    """
    def __init__(self,
                 gradient_descent=False,
                 least_square=False,
                 mle=False,
                 lr=0.0001,
                 precision=0.001,
                 max_iters=1000,
                 previous_step_size=1):
        self.gradient_descent = gradient_descent
        self.least_square = least_square
        self.mle = mle
        self.lr = lr
        self.precision = precision
        self.max_iters = max_iters
        self.previous_step_size = previous_step_size

    def compute_points_error(self,points, b, m):
        """
        y = m*x+b
        m is slope, b is y-intercept
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
            # x = points[0][i]
            # y = points[1][i]
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
        b =  0
        m = 0
        while iters < self.max_iters:
            b, m = self.__step_gradient(x, y, b, m)
            iters += 1
        return [b, m]

    def __least_square(self, x, y):
        """
        given training values for x and y,
        find the least-squares values of alpha and beta
        """
        beta = Stats().correlation(x, y) * Stats().standard_deviation(y) / \
            Stats().standard_deviation(x)
        alpha = Stats().mean(y) - beta * Stats().mean(x)
        return [alpha, beta]
    
    def __L(self, x, y):        
        raise NotImplementedError

    def fit(self, x, y):
        if self.gradient_descent:
            b, m = self.__run_step_gradient(x, y)
            return [b, m]
        elif self.least_square:
            b, m = self.__least_square(x, y)
            return [b, m]
        elif self.mle:
            b, m = self.__L(x, y)
