import numpy as np
import matplotlib.pyplot as plt
from .loss_func import Loss_Func

def Coefficient_of_Determination(predict, actual, scaler_y):
    # 역정규화
    predict = scaler_y.inverse_transform(predict)
    actual = scaler_y.inverse_transform(actual)

    actual_mean = np.mean(actual)  # 실제값의 평균

    SSE = np.sum((actual - predict) ** 2)
    SST = np.sum((actual - actual_mean) ** 2)
    R = 1 - (SSE / SST)

    return R

def plot_actual_vs_pred(predict, actual, scaler_y):
    # 역정규화
    predict = scaler_y.inverse_transform(predict)
    actual = scaler_y.inverse_transform(actual)

    # 그래프 그리기
    plt.figure(figsize=(12, 4))

    min_val = min(min(predict), min(actual))
    max_val = max(max(predict), max(actual))
    plt.plot([min_val, max_val], [min_val, max_val], 'k-', label='Actual')

    plt.scatter(predict, actual, s=2, label='Predicted')

    plt.title("Actual vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.legend()
    plt.show()

def plot_residual_vs_pred(predict, actual, scaler_y):
    # 역정규화
    predict = scaler_y.inverse_transform(predict)
    actual = scaler_y.inverse_transform(actual)

    # residual 계산
    residuals = actual - predict

    # 그래프 그리기
    plt.figure(figsize=(8, 5))
    plt.scatter(predict, residuals, s=1)

    plt.axhline(y=0, color='black', linestyle='--', lw=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual by Predicted Plot")
    plt.show()

def Compute_MFE_MRE(predict_train, actual_train, predict_test, actual_test, scaler_y):
    # 역정규화
    predict_train = scaler_y.inverse_transform(predict_train)
    actual_train = scaler_y.inverse_transform(actual_train)
    predict_test = scaler_y.inverse_transform(predict_test)
    actual_test = scaler_y.inverse_transform(actual_test)

    # MFE (훈련 데이터에서 모델이 얼마나 잘 맞는지)
    MFE = Loss_Func().MAE(predict_train, actual_train)

    # MRE (새로운 데이터에서도 모델이 잘 맞는지)
    MRE = Loss_Func().MAE(predict_test, actual_test)

    print("MFE (Model Fit Error): ", MFE)
    print("MRE (Model Representation Error): ", MRE) 