# Importez les données de data1.csv et testez une régression
# linéaire entre la longueur et l’épaisseur des pétales.

import pandas as pd
from os import path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv(path.join(path.dirname(__file__), path.normpath('../../data/scikit-learn/data1.csv')))

X = df[['PetalLength']].iloc[:, :].values
y = df[['PetalWidth']].iloc[:, :].values

reg = LinearRegression()
reg.fit(X, y)

pred = reg.predict(X)

plt.scatter(X, y,c='orange')
plt.plot(X, pred)

plt.xlabel('Epaisseur des Pétales')
plt.ylabel('Longueur des Pétales')
plt.show()