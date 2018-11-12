import random

import pandas as pd
import numpy as np

from gradient_descent import minimize_stochastic, safe

class MultiReg:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.beta = None

    def dot(self, v, w):
        """v_1 * w_1 + ... + v_n * w_n
        """
        return sum(v_i * w_i for v_i, w_i in zip(v, w))

    def predict(self, x_i, beta):
        return self.dot(x_i, beta)

    def error(self, x_i, y_i, beta):
        return y_i - self.predict(x_i, beta)

    def squared_error(self, x_i, y_i, beta):
        return self.error(x_i, y_i, beta) ** 2

    def squared_error_gradient(self, x_i, y_i, beta):
        """the gradient corresponding to the ith squared error term
        """
        return [-2 * x_ij * self.error(x_i, y_i, beta) 
                for x_ij in x_i]

    def fit(self, x: np.ndarray, y: np.array):
        beta_initial = [random.random() for x_i in x[0]]
        print(beta_initial)
        self.beta = minimize_stochastic(safe(self.squared_error), 
                                safe(self.squared_error_gradient),
                                x, y,   
                                beta_initial, 
                                self.learning_rate)

    def regularization(self):
        # TODO: regularization part
        raise NotImplementedError
