# Importez les données de data2.csv, gardez uniquement les
# champs adm, dip et mil et entraînez une ACP dessus.

# Comparez graphiquement les représentations des données
# utilisant deux axes standards (via une scatter matrix) et celles
# utilisant les axes de l’ACP.

import pandas as pd
from os import path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv(path.join(path.dirname(__file__), path.normpath('../../data/scikit-learn/data2.csv')), sep=';')
df = df[['adm', 'dip', 'mil']].dropna()

pca = PCA()
rot = pca.fit_transform(df)

pd.plotting.scatter_matrix(df)
plt.show()

plt.scatter(rot[:,0], rot[:,1], alpha=0.4)
plt.show()