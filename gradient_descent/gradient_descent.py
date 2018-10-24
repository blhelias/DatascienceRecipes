# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 22:42:03 2018

@author: brieuc
"""
from typing import Callable, List, Optional
import numpy as np
import random
import math


def minimize_stochastic(target_fn: Callable,
                        gradient_fn: Callable,
                        x: List[List[float]], y: List[float],
                        theta_0: List[float],
                        learning_rate: float) -> Optional[List[float]]:
    num_iters_with_no_improvements = 0
    data = list(zip(x, y))
    theta = theta_0
    min_theta, min_value = None, float("inf")
    lr = learning_rate
    count=0
    while num_iters_with_no_improvements < 100:
        loss = (1/2*len(x)) * sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)
#        print("loss", loss)
        if loss < min_value:
            improvement = min_value - loss
            if improvement <= 0.01:
                count += 1
                if count >= 100:
                    return min_theta
            else:
                count = 0
                
            min_theta, min_value = theta, loss
            num_iters_with_no_improvements = 0
            lr = learning_rate
        else:
            # reduce the learning rate when no improvement on loss
            num_iters_with_no_improvements += 1
            lr *= 0.9
        # update value with descent gradient
        for x_i, y_i in random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_substract(theta, scalar_multiply(lr, gradient_i))
    return min_theta

def safe(f):
    """define a new function that wraps f and return it"""
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')         # this means "infinity" in Python
    return safe_f

def scalar_multiply(scalar, vector):
    # vector = map(lambda x: x * scalar, vector)
    # return list(vector)
#    print("scalar", scalar)
#    print("vector", vector)
    return [scalar * v_i for v_i in vector]

def vector_substract(vec1, vec2):
    # sub = map(float.__sub__, vec1, vec2)
    # return list(sub)
    return [v_i - w_i for v_i, w_i in zip(vec1,vec2)]
def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))
    
def random_order(arr: np.ndarray):
    """generator that returns the elements of data in random order"""
    indexes: List[int] = [i for i, _ in enumerate(arr)]
    random.shuffle(indexes)
    for i in indexes:
        yield arr[i]