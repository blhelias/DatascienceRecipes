"""
Write a program that prints the numbers from 1 to 100. 
But for multiples of three print "Fizz" 
instead of the number and for the multiples of five print "Buzz".
For numbers which are multiples of both three and five print "FizzBuzz".

using neural Net, we can see the problem as depending on the number, we output different class
"""

from typing import List

import numpy as np

from brieucNet.train import train
from brieucNet.nn import NeuralNet
from brieucNet.layers import Linear, Tanh
from brieucNet.optim import SGD

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 ==0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_encode(x: int) -> List[int]:
    """
    10 digits binary encoding of x
    """
    return [x >> i & 1 for i in range(10)]

inputs = np.array([
    binary_encode(x)
    for x in range(101, 1024)
])

targets = np.array([
    fizz_buzz_encode(x)
    for x in range(101, 1024)
])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])

train(net,
      inputs,
      targets,
      num_epoch=5000,
      optimizer=SGD(lr=0.001))

for x in range(1, 101):
    predicted = net.forward(binary_encode(x))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = (str(x), "Fizz", "Buzz", "FizzBuzz")
    print(x, labels[predicted_idx], labels[actual_idx])

