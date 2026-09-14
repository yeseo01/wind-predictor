"""Training utilities for the wind-prediction model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from evaluation.loss_func import LossFunction


def train_model(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 0.001,
    epochs: int = 1000,
    batch_size: int = 16,
) -> dict[str, list[float]]:
    """Train the model with mini-batch gradient descent.

    Args:
        model: Model object providing ``forward`` and ``backward`` methods.
        x_train: Training input features.
        y_train: Training target values.
        learning_rate: Learning rate used during backpropagation.
        epochs: Number of training epochs.
        batch_size: Number of samples per mini-batch.

    Returns:
        A history dictionary containing per-epoch training loss.
    """
    history: dict[str, list[float]] = {"loss": []}
    loss_fn = LossFunction()

    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0

        for start_idx in range(0, len(x_train), batch_size):
            end_idx = start_idx + batch_size
            batch_x = x_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]

            predictions = model.forward(batch_x)
            loss = loss_fn.mse(predictions, batch_y)

            total_loss += loss
            batch_count += 1

            grad = loss_fn.gradient(predictions, batch_y)
            model.backward(grad, learning_rate)

        avg_loss = total_loss / batch_count

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        history["loss"].append(avg_loss)

    return history


def plot_training_history(history: dict[str, list[float]]) -> None:
    """Plot the training-loss history."""
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
