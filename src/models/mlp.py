from .layer import Layer

class Multi_Layer_Perceptron():
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = [] # 설계한 MLP에 존재하는 layer class들을 저장할 리스트

        # 입력층
        self.layers.append(Layer(input_size, hidden_sizes[0]))

        # 은닉층들
        for i in range(len(hidden_sizes)-1):
            self.layers.append(Layer(hidden_sizes[i], hidden_sizes[i+1]))

        # 출력층
        self.layers.append(Layer(hidden_sizes[-1], output_size))

    def forward(self, x):
        # Forward
        for layer in self.layers[:-1]:  # 마지막 레이어(출력층)를 제외한 모든 레이어
            x = layer.forward(x)
            x = layer.relu(x)

        # 출력층
        x = self.layers[-1].forward(x)
        return x

    def backward(self, grad, learning_rate):
        # Backward
        for layer in reversed(self.layers):  # 출력층부터 입력층까지 역순으로
            grad = layer.backward(grad, learning_rate) 