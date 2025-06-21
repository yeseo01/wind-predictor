# Wind Predictor MLP

A Multi-Layer Perceptron (MLP)-based regression model that predicts eastward wind speed using geographic coordinates (latitude and longitude) as input.

## 📌 Project Overview
This project was part of an assignment in Spring 2025, aimed at:
- Implementing a basic MLP from scratch
- Applying it to a real-world wind dataset
- Evaluating model performance using standard regression metrics and plots

## 📁 Dataset
- **Source**: US Weather Data
- **Features**: Latitude, Longitude
- **Target**: Eastward wind speed

## 🧠 Model
- Architecture: Fully connected MLP
- Training method: Backpropagation
- Evaluation: R² score, residual plot, actual vs. predicted plot

## 📊 Results
- R² Score: 0.9790
- MFE: 1.6484
- MRE: 1.7230
- Visualizations: See `results/` folder

## 🚀 How to Run
```bash
pipenv run python src/main.py
