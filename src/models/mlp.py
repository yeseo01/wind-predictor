"""Multi-layer perceptron implemented from scratch with NumPy."""

from __future__ import annotations

import numpy as np

from .layer import Layer


class MultiLayerPerceptron:
    """Feed-forward neural network composed of dense layers."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
    ) -> None:
        self.layers: list[Layer] = []

        self.layers.append(Layer(input_size, hidden_sizes[0]))

        for i in range(len(hidden_sizes) - 1):
            self.layers.append(
                Layer(hidden_sizes[i], hidden_sizes[i + 1])
            )

        self.layers.append(Layer(hidden_sizes[-1], output_size))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run a forward pass through the network."""
        for layer in self.layers[:-1]:
            x = layer.forward(x)
            x = layer.relu(x)

        x = self.layers[-1].forward(x)
        return x

    def backward(
        self,
        grad: np.ndarray,
        learning_rate: float,
    ) -> None:
        """Backpropagate gradients through all layers."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
