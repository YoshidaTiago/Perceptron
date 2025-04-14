import numpy as np
from sklearn.linear_model import Perceptron

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

epochs = int(input("Quantas vezes deseja iterar pelo conjunto? "))

p = Perceptron(max_iter=epochs, eta0=0.1, fit_intercept=True, tol=None)

p.fit(X, y)

print(f"w1 = {p.coef_[0][0]}, w2 = {p.coef_[0][1]}, bias = {p.intercept_[0]}")

for x in X:
    print(f"Entrada: {x} -> Saída: {p.predict([x])[0]}")
