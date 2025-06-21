import numpy as np

class Layer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # he 초기화 사용
        self.weight = np.random.normal(0, np.sqrt(2.0 / input_size), (output_size, input_size))
        self.bias = np.zeros((output_size, 1))

        self.weight_grad = np.zeros_like(self.weight)  # 가중치에 대한 손실함수의 기울기
        self.bias_grad = np.zeros_like(self.bias)      # 바이어스에 대한 손실함수의 기울기
        self.input = None
        self.output = None
        self.activated = None   # 활성화함수 값을 저장

    def forward(self, x):
        self.input = x
        self.output = np.dot(self.weight, x.T).T + self.bias.T
        return self.output

    def backward(self, grad, learning_rate):
        # ReLU의 기울기 적용
        if self.activated is not None:
            grad = grad * (self.activated > 0)

        # 가중치와 바이어스의 기울기 계산
        self.weight_grad = np.dot(grad.T, self.input)
        self.bias_grad = np.sum(grad, axis=0, keepdims=True).T

        # 다음 레이어로 전파할 기울기 계산
        next_grad = np.dot(grad, self.weight)

        # 파라미터 업데이트
        self.weight -= learning_rate * self.weight_grad
        self.bias -= learning_rate * self.bias_grad

        return next_grad

    def relu(self, x):
        self.activated = np.maximum(0, x)
        return self.activated 