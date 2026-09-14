"""Evaluation utilities for the wind-prediction model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .loss_func import LossFunction


def coefficient_of_determination(
    predict: np.ndarray,
    actual: np.ndarray,
    scaler_y,
) -> float:
    """Compute R-squared in the original target scale."""
    predict = scaler_y.inverse_transform(predict)
    actual = scaler_y.inverse_transform(actual)

    actual_mean = np.mean(actual)

    sse = np.sum((actual - predict) ** 2)
    sst = np.sum((actual - actual_mean) ** 2)
    r2 = 1 - (sse / sst)

    return float(r2)


def plot_actual_vs_pred(
    predict: np.ndarray,
    actual: np.ndarray,
    scaler_y,
) -> None:
    """Plot actual target values against model predictions."""
    predict = scaler_y.inverse_transform(predict)
    actual = scaler_y.inverse_transform(actual)

    plt.figure(figsize=(12, 4))

    min_val = min(np.min(predict), np.min(actual))
    max_val = max(np.max(predict), np.max(actual))

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k-",
        label="Actual",
    )
    plt.scatter(
        predict,
        actual,
        s=2,
        label="Predicted",
    )

    plt.title("Actual vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.legend()
    plt.show()


def plot_residual_vs_pred(
    predict: np.ndarray,
    actual: np.ndarray,
    scaler_y,
) -> None:
    """Plot residuals against predicted values."""
    predict = scaler_y.inverse_transform(predict)
    actual = scaler_y.inverse_transform(actual)

    residuals = actual - predict

    plt.figure(figsize=(8, 5))
    plt.scatter(predict, residuals, s=1)

    plt.axhline(
        y=0,
        color="black",
        linestyle="--",
        lw=1,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual vs Predicted")
    plt.show()


def compute_mfe_mre(
    predict_train: np.ndarray,
    actual_train: np.ndarray,
    predict_test: np.ndarray,
    actual_test: np.ndarray,
    scaler_y,
) -> None:
    """Compute MAE-based fit and representation errors."""
    predict_train = scaler_y.inverse_transform(predict_train)
    actual_train = scaler_y.inverse_transform(actual_train)
    predict_test = scaler_y.inverse_transform(predict_test)
    actual_test = scaler_y.inverse_transform(actual_test)

    loss_fn = LossFunction()

    mfe = loss_fn.mae(predict_train, actual_train)
    mre = loss_fn.mae(predict_test, actual_test)

    print("MFE (Model Fit Error): ", mfe)
    print("MRE (Model Representation Error): ", mre)
