# Même question, cette fois entre la longueur des cépales et la
# largeur des pétales. Comparez les scores des deux
# régressions.
# Score: (ici) 0.67 vs 0.93

import pandas as pd
from os import path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv(path.join(path.dirname(__file__), path.normpath('../../data/scikit-learn/data1.csv')))

X = df[['SepalLength']].iloc[:, :].values
y = df[['PetalWidth']].iloc[:, :].values

reg = LinearRegression()
reg.fit(X, y)

print(reg.score(X, y))

pred = reg.predict(X)

plt.scatter(X, y,c='orange')
plt.plot(X, pred)

plt.xlabel('Epaisseur des Pétales')
plt.ylabel('Longueur des Sépales')
plt.show()