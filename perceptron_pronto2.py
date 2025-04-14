import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron

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

def train_and_test(X_train, y_train, X_test, y_test, label):
    clf = Perceptron(max_iter=1000, eta0=0.1, random_state=42, tol=None)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test) * 100
    print(f"Precisão no conjunto de teste para {label}: {accuracy:.2f}%")

train_and_test(X_train, y_setosa_train, X_test, y_setosa_test, 'Setosa')
train_and_test(X_train, y_versicolor_train, X_test, y_versicolor_test, 'Versicolor')
train_and_test(X_train, y_virginica_train, X_test, y_virginica_test, 'Virginica')