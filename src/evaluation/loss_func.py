"""Loss functions used for training and evaluation."""

from __future__ import annotations

import numpy as np


class LossFunction:
    """Collection of loss functions and gradients."""

    def mse(
        self,
        predict: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Compute half mean squared error."""
        return float(np.mean((y - predict) ** 2) / 2)

    def mae(
        self,
        predict: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Compute mean absolute error."""
        return float(np.mean(np.abs(y - predict)))

    def gradient(
        self,
        predict: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Compute the gradient of half mean squared error."""
        return (predict - y) / len(y)
