# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 23:28:52 2018

@author: brieu
"""
import numpy as np

class LogisticRegression:
    """
    Let's build a logistic regression model
    
    1) plutot que de prédire une classe, on prédit la probabilité d'appartenir
        a la classe 1.
        
        p(y = 1|x) = hw(x) = logistic(w.x)
    2) choisir la classe la plus propbable selon le model:
        Si hw(x) >= 0.5:
            choisir classe 1
        Sinon:
            choisir classe 0
            
    3) on définit une fonction de cout:
        L(y, hw(x)) = - y * log(hw(x)) - (1 - y) * log(1 - hw(x)))
    
        on veut donc minimiser (y - hw(x))
    """
    def __init__(self):
        pass
    
    def sigmoid(self, wx: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-wx))
    

    def minimize_stochastic(self, target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
        data = zip(x, y)
        theta = theta_0 # initial guess
        alpha = alpha_0 # initial step size
        min_theta, min_value = None, float("inf") # the minimum so far
        iterations_with_no_improvement = 0
        # if we ever go 100 iterations with no improvement, stop
        while iterations_with_no_improvement < 100:
            value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data )
            if value < min_value:
                # if we've found a new minimum, remember it
                # and go back to the original step size
                min_theta, min_value = theta, value
                iterations_with_no_improvement = 0
                alpha = alpha_0
            else:
                # otherwise we're not improving, so try shrinking the step size
                iterations_with_no_improvement += 1
                alpha *= 0.9
                # and take a gradient step for each of the data points
            for x_i, y_i in in_random_order(data):
                gradient_i = gradient_fn(x_i, y_i, theta)
                theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
        return min_theta
        
def in_random_order(data: np.ndarray) -> np.ndarray:
    raise NotImplementedError

def vector_subtract(vec1, vec2):
    raise NotImplementedError

def scalar_multiply(sca1, sca2):
    raise NotImplementedError   
