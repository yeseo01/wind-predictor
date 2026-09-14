# Wind Predictor MLP

A Multi-Layer Perceptron (MLP) regression model that predicts eastward wind speed from geographic coordinates.

The neural network is implemented from scratch with NumPy, including dense layers, forward propagation, ReLU activation, backpropagation, loss computation, and parameter updates. No deep-learning framework such as TensorFlow or PyTorch is used.

This project was originally developed as part of a Spring 2025 aerospace AI coursework assignment and has since been cleaned up for reproducibility and readability.

## Project Overview

The model predicts eastward wind speed using two input features:

- Longitude
- Latitude

The main purpose of the project is to implement and understand the mechanics of a neural network without relying on a high-level deep-learning framework.

Supporting libraries are used for data loading, visualization, and train/test splitting, while the MLP itself is implemented directly with NumPy.

## Model Architecture

The network uses the following architecture:

```text
Input (2)
   │
   ▼
Dense (16) + ReLU
   │
   ▼
Dense (16) + ReLU
   │
   ▼
Dense (1)
   │
   ▼
Eastward Wind Prediction
```

The implemented neural-network components include:

- Fully connected layers
- He weight initialization
- ReLU activation
- Forward propagation
- Backpropagation
- Mini-batch gradient descent
- Mean squared error loss
- Mean absolute error
- Custom standardization and inverse transformation

The current architecture is:

```text
2 → 16 → 16 → 1
```

## Dataset

The dataset is stored at:

```text
data/20190120_Time_8_Altitude_22_Eastward_wind.csv
```

### Input Features

- `Longitude (deg)`
- `Latitude (deg)`

### Target

- `Eastward wind (m/s)`

The original coursework described the dataset as US weather data. The exact upstream source and licensing information are not currently documented in this repository.

## Data Preprocessing

The dataset is split into training and test sets using:

- Training set: 80%
- Test set: 20%
- `random_state=42`

A custom standardization class is used to normalize the input features and target values.

To prevent data leakage, the scaler statistics are calculated using the training set only. The same fitted statistics are then used to transform both the training and test sets.

## Training

The current training configuration is:

| Parameter | Value |
| --- | ---: |
| Hidden layers | 2 |
| Hidden units | 16, 16 |
| Learning rate | 0.01 |
| Epochs | 1000 |
| Batch size | 16 |
| Train/test split | 80/20 |
| Random seed | 42 |

The network is trained using mini-batch gradient descent with manually implemented backpropagation.

## Results

Using the current preprocessing and training pipeline:

| Metric | Result |
| --- | ---: |
| R² | **0.9755** |
| MFE | **1.7327 m/s** |
| MRE | **1.8288 m/s** |

The R² score is evaluated on the test set after converting the standardized predictions back to the original wind-speed scale.

In this project:

- **MFE (Model Fit Error)** is the mean absolute error on the training set.
- **MRE (Model Representation Error)** is the mean absolute error on the test set.

These names follow the terminology used in the original coursework. In the current implementation, both metrics are MAE-based quantities.

The program also generates:

- Training loss history
- Actual vs. predicted plot
- Residual vs. predicted plot

## Repository Structure

```text
wind-predictor/
├── data/
│   └── 20190120_Time_8_Altitude_22_Eastward_wind.csv
├── src/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluation.py
│   │   └── loss_func.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── layer.py
│   │   └── mlp.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── scaler.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py
│   └── main.py
├── .gitignore
├── Pipfile
├── Pipfile.lock
└── README.md
```

## How to Run

### 1. Install Pipenv

If Pipenv is not already installed:

```bash
python3 -m pip install pipenv
```

### 2. Install Dependencies

From the repository root:

```bash
pipenv install
```

### 3. Run the Model

```bash
pipenv run python src/main.py
```

The script trains the MLP, prints the evaluation metrics, and displays the training and evaluation plots.

## Dependencies

The project uses:

- **NumPy** — neural-network implementation and numerical computation
- **pandas** — CSV data loading
- **scikit-learn** — train/test splitting
- **Matplotlib** — visualization

The neural network itself does not depend on TensorFlow, PyTorch, or another deep-learning framework.
