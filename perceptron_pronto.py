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

clf = Perceptron(max_iter=epochs, eta0=0.1, fit_intercept=True, tol=None)

clf.fit(X, y)

print(f"w1 = {clf.coef_[0][0]}, w2 = {clf.coef_[0][1]}, bias = {clf.intercept_[0]}")

for x in X:
    print(f"Entrada: {x} -> Saída: {clf.predict([x])[0]}")
