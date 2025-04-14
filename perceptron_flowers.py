import pandas as pd
import numpy as np

df = pd.read_csv('Perceptron\\docs\\iris.csv')

X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values

y_setosa = df['variety'].map({'Setosa': 1, 'Versicolor': 0, 'Virginica': 0}).values
y_versicolor = df['variety'].map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 0}).values
y_virginica = df['variety'].map({'Setosa': 0, 'Versicolor': 0, 'Virginica': 1}).values

indices = np.random.permutation(len(X))
train_size = float(input("Escreva a quantidade que deseja utilizar para o conjunto teste (0 a 1): "))
train_size = int(len(X) * train_size)

X_train = X[indices[:train_size]]
X_test = X[indices[train_size:]]

y_setosa_train = y_setosa[indices[:train_size]]
y_setosa_test = y_setosa[indices[train_size:]]

y_versicolor_train = y_versicolor[indices[:train_size]]
y_versicolor_test = y_versicolor[indices[train_size:]]

y_virginica_train = y_virginica[indices[:train_size]]
y_virginica_test = y_virginica[indices[train_size:]]

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate

    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.activation(z)
    
    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error
    
    def test(self, X, y):
        correct = 0
        for xi, target in zip(X, y):
            prediction = self.predict(xi)
            if prediction == target:
                correct += 1
        accuracy = (correct / len(y)) * 100
        return accuracy

p_setosa = Perceptron(input_size=4)
p_setosa.train(X_train, y_setosa_train)
accuracy_setosa = p_setosa.test(X_test, y_setosa_test)
print(f"Precisão no conjunto de teste para Setosa: {accuracy_setosa:.2f}%")

p_versicolor = Perceptron(input_size=4)
p_versicolor.train(X_train, y_versicolor_train)
accuracy_versicolor = p_versicolor.test(X_test, y_versicolor_test)
print(f"Precisão no conjunto de teste para Versicolor: {accuracy_versicolor:.2f}%")

p_virginica = Perceptron(input_size=4)
p_virginica.train(X_train, y_virginica_train)
accuracy_virginica = p_virginica.test(X_test, y_virginica_test)
print(f"Precisão no conjunto de teste para Virginica: {accuracy_virginica:.2f}%")