import numpy as np

class Loss_Func():
    def __init__(self):
        pass

    # 평균제곱오차
    def MSE(self, predict, y):
        return np.mean((y - predict) ** 2) / 2

    # 평균절대오차
    def MAE(self, predict, y):
        return np.mean(np.abs(y - predict))

    # 평균제곱오차 미분값
    def gradient(self, predict, y):
        return (predict - y) / len(y) 