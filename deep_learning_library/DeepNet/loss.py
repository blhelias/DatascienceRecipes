"""
loss function measures how good our predictions are,
we can use this to adjust our parameters

TODO:
add some crossentropy loss, regularization term ...
"""
import numpy as np

from brieucNet.tensor import Tensor


class Loss:
    """
    MSE is mean squared error, although were
    just going to do total squared error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """
    MSE is mean squared error, although were
    just going to do total squared error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)
