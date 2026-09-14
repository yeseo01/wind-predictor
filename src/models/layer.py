"""Dense layer implementation used by the MLP."""

from __future__ import annotations

import numpy as np


class Layer:
    """Fully connected layer with optional ReLU activation."""

    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.output_size = output_size

        # He initialization
        self.weight = np.random.normal(
            0,
            np.sqrt(2.0 / input_size),
            (output_size, input_size),
        )
        self.bias = np.zeros((output_size, 1))

        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)

        self.input: np.ndarray | None = None
        self.output: np.ndarray | None = None
        self.activated: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the linear transformation for this layer."""
        self.input = x
        self.output = np.dot(self.weight, x.T).T + self.bias.T
        return self.output

    def backward(
        self,
        grad: np.ndarray,
        learning_rate: float,
    ) -> np.ndarray:
        """Backpropagate gradients and update layer parameters."""
        if self.activated is not None:
            grad = grad * (self.activated > 0)

        self.weight_grad = np.dot(grad.T, self.input)
        self.bias_grad = np.sum(grad, axis=0, keepdims=True).T

        next_grad = np.dot(grad, self.weight)

        self.weight -= learning_rate * self.weight_grad
        self.bias -= learning_rate * self.bias_grad

        return next_grad

    def relu(self, x: np.ndarray) -> np.ndarray:
        """Apply the ReLU activation function."""
        self.activated = np.maximum(0, x)
        return self.activated
