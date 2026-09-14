"""Standardization utilities for model inputs and targets."""

from __future__ import annotations

import numpy as np


class Scaler:
    """Standardize data using the mean and standard deviation."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Scaler":
        """Compute the mean and standard deviation of the input data."""
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

        # Avoid division by zero for constant features
        self.std[self.std == 0] = 1

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Standardize data using the fitted statistics."""
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Transform standardized data back to its original scale."""
        return x * self.std + self.mean
