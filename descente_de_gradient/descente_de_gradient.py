# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 22:42:03 2018

@author: brieuc
"""
from typing import Callable, List, Optional
import numpy as np
import random



def minimize_stochastic(target_fn: Callable,
                        gradient_fn: Callable,
                        x: List[List[float]], y: List[float],
                        theta_0: List[float],
                        learning_rate: float = 0.01) -> Optional[List[float]]:
    num_iters_with_no_improvements = 0
    data = list(zip(x, y))
    theta = theta_0
    min_theta, min_value = None, float("inf")
    alpha = learning_rate
    while num_iters_with_no_improvements < 100:
        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data)
        if value < min_value:
            min_theta, min_value = theta, value
            num_iters_with_no_improvements = 0
            alpha = learning_rate
        else:
            num_iters_with_no_improvements += 1
            alpha *= 0.9
        # update value
        for x_i, y_i in random_order(data):
            gradient_i = gradient_fn(x_i, y_i,theta)
            theta = vector_substract(theta, scalar_multiply(alpha, gradient_i))
        print(min_theta)
    return min_theta

def scalar_multiply(scalar, vector):
    vector = map(lambda x: x * scalar, vector)
    return list(vector)

def vector_substract(vec1, vec2):
    sub = map(float.__sub__, vec1, vec2)
    return list(sub)

def random_order(arr: np.ndarray):
    """generator that returns the elements of data in random order"""
    indexes: List[int] = [i for i, _ in enumerate(arr)]
    random.seed(0)
    random.shuffle(indexes)
    for i in indexes:
        yield arr[i]
    
    


   