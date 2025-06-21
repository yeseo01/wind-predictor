import numpy as np

class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        # 평균과 표준편차 계산
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # 표준편차가 0인 경우 1로 설정 (transform에서 0으로 나누는 것을 방지)
        self.std[self.std == 0] = 1
        return self

    def transform(self, X):
        # 정규화 수행: (X - mean) / std
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        # 역변환 수행: X * std + mean
        return X * self.std + self.mean 