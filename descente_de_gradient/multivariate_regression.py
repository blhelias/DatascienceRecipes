import random

import pandas as pd
import numpy as np

from descente_de_gradient import minimize_stochastic

df = pd.read_csv("data/Sales.csv", sep=";")
x = df[df.columns[2:]].values

y = df[df.columns[1]].values
y = map(lambda x: x.replace(",", "."), y)
y = list(map(float, y))

def predict(x_i, beta):
    return np.dot(x_i, beta)

def error(x_i, y_i, beta):
    return y_i - predict(x_i, beta)

def squared_error(x_i, y_i, beta):
    return error(x_i, y_i, beta) ** 2

def squared_error_gradient(x_i, y_i, beta):
    """the gradient corresponding to the ith squared error term"""
    
    return [-2 * x_ij * error(x_i, y_i, beta) for x_ij in x_i]

def estimate_beta(x, y):
    beta_initial = [random.random() for x_i in x[0]]
    print(beta_initial)
    return minimize_stochastic(squared_error, 
                               squared_error_gradient, 
                               x, y, 
                               beta_initial, 
                               0.001)

print(estimate_beta(x, y))
