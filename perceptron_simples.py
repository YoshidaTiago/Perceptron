import numpy as np

class Perceptron:
    def __init__(self, input_size, epochs, learning_rate = 0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.activation(z)
    
    def train(self, X, y):
        for epoch in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction

                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error
                total_error = abs(error)
            if total_error == 0:
                break
            else:
                print(f"w1 = {self.weights[0]}, w2 = {self.weights[1]}, bias = {self.bias}")

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

epochs = int(input("Quantas vezes deseja iterar pelo conjunto? "))

p = Perceptron(input_size=2, epochs=epochs)
p.train(X, y)

for x in X:
    print(f"Entrada: {x} -> Saída: {p.predict(x)}")