import numpy as np
import matplotlib.pyplot as plt
from Test.loss_func import Loss_Func

def Train(model, x_train, y_train, learning_rate=0.001, epochs=1000, batch_size=16):
    history = { 'loss': [] }

    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0

        # 배치 크기로 학습
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size] # 실제 값

            # Forward
            predict = model.forward(batch_x)

            # 해당 배치의 평균 손실 계산 후 저장
            loss = Loss_Func().MSE(predict, batch_y)

            total_loss += loss
            batch_count += 1

            # Backward
            grad = Loss_Func().gradient(predict, batch_y) # 손실함수의 기울기 계산
            model.backward(grad, learning_rate)

        # 전체 데이터의 평균 손실 계산
        avg_loss = total_loss / batch_count

        # 일정 에폭마다 평균 손실 출력
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

        # 에폭별로 평균 손실값 저장
        history['loss'].append(avg_loss)

    return history

def plot_training_history(history):
    # 손실 그래프 그리기
    plt.figure(figsize=(8, 5))

    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout() # 여백 자동 조절
    plt.show() 