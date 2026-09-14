from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing.scaler import Scaler
from models.mlp import MultiLayerPerceptron
from training import train_model, plot_training_history
from evaluation import (
    coefficient_of_determination,
    plot_actual_vs_pred,
    plot_residual_vs_pred,
    compute_mfe_mre
)

# 랜덤 시드 설정
np.random.seed(42)

# ----------------- 데이터 처리 --------------------
# 데이터 가져오기
repo_root = Path(__file__).resolve().parent.parent
data_path = repo_root / "20190120_Time_8_Altitude_22_Eastward_wind.csv"

df = pd.read_csv(data_path)
x = df[['Longitude (deg)', 'Latitude (deg)']].values
y = df['Eastward wind (m/s)'].values.reshape(-1, 1)

# 학습/테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
)

# 학습 데이터로만 정규화 기준 계산
scaler_x = Scaler().fit(x_train)
scaler_y = Scaler().fit(y_train)

x_train = scaler_x.transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)

# ------------------ 모델 생성 -------------------
# 모델 초기화
input_size = 2
hidden_sizes = [16, 16]
output_size = 1
model = MultiLayerPerceptron(input_size, hidden_sizes, output_size)

# 학습 파라미터 설정
params = { 'learning_rate': 0.01, 'epochs': 1000, 'batch_size': 16}

# ------------------- 모델 학습 --------------------
print("=== 학습 시작 ===")
history = train_model(model, x_train, y_train, **params)

# 학습 과정 시각화
plot_training_history(history)

# -------------------- 모델 평가 --------------------
print("\n=== 테스트 결과 ===")
predict_train = model.forward(x_train)
predict_test = model.forward(x_test)

# STEP 1: 결정계수
R2 = coefficient_of_determination(predict_test, y_test, scaler_y)
print("R2: ", R2)

# STEP 2: Actual by predicted plot
plot_actual_vs_pred(predict_test, y_test, scaler_y)

# STEP 3: Residual by predicted plot
plot_residual_vs_pred(predict_test, y_test, scaler_y)

# STEP 4: MFE, MRE계산
compute_mfe_mre(predict_train, y_train, predict_test, y_test, scaler_y)
