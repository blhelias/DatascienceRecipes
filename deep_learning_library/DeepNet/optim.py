"""
we use an optimizer to adjust 
params of ou network based on the gradient computed
during backproagation
TODO:
add some other optimizers !
"""

from brieucNet.nn import NeuralNet

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net:NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad