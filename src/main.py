from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from preprocessing.scaler import Scaler
from models.mlp import MultiLayerPerceptron
from training import train_model, plot_training_history
from evaluation import (
    coefficient_of_determination,
    plot_actual_vs_pred,
    plot_residual_vs_pred,
    compute_mfe_mre,
)


# Set random seed
np.random.seed(42)


# -------------------- Data Processing --------------------

# Load dataset
repo_root = Path(__file__).resolve().parent.parent
data_path = (
    repo_root
    / "data"
    / "20190120_Time_8_Altitude_22_Eastward_wind.csv"
)

df = pd.read_csv(data_path)

x = df[["Longitude (deg)", "Latitude (deg)"]].values
y = df["Eastward wind (m/s)"].values.reshape(-1, 1)


# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
)


# Fit scalers on training data only
scaler_x = Scaler().fit(x_train)
scaler_y = Scaler().fit(y_train)

x_train = scaler_x.transform(x_train)
x_test = scaler_x.transform(x_test)

y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)


# -------------------- Model Setup --------------------

# Initialize model
input_size = 2
hidden_sizes = [16, 16]
output_size = 1

model = MultiLayerPerceptron(
    input_size,
    hidden_sizes,
    output_size,
)


# Set training parameters
params = {
    "learning_rate": 0.01,
    "epochs": 1000,
    "batch_size": 16,
}


# -------------------- Model Training --------------------

print("=== Training ===")

history = train_model(
    model,
    x_train,
    y_train,
    **params,
)

# Visualize training history
plot_training_history(history)


# -------------------- Model Evaluation --------------------

print("\n=== Evaluation Results ===")

predict_train = model.forward(x_train)
predict_test = model.forward(x_test)


# STEP 1: R-squared
r2 = coefficient_of_determination(
    predict_test,
    y_test,
    scaler_y,
)

print("R2: ", r2)


# STEP 2: Actual vs. predicted values
plot_actual_vs_pred(
    predict_test,
    y_test,
    scaler_y,
)


# STEP 3: Residuals vs. predicted values
plot_residual_vs_pred(
    predict_test,
    y_test,
    scaler_y,
)


# STEP 4: Compute MFE and MRE
compute_mfe_mre(
    predict_train,
    y_train,
    predict_test,
    y_test,
    scaler_y,
)

plt.show()
