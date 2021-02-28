# Importez les données iris avec dataset.load_iris. Effectuez
# un k-means avec 2,3,4 valeurs. Comparez graphiquement les
# résultats avec les valeurs cibles.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
data = np.delete(iris.data, slice(-2, None), 1) # On garde que les deux premières colonnes
x, y = zip(*data)
plt.subplot(221)
plt.scatter(x, y, c=iris.target, alpha=0.7)
plt.title("Données")

# k = 2
km2 = KMeans(2).fit(data)
plt.subplot(222)
plt.scatter(x, y, c=km2.labels_, alpha=0.7)
plt.scatter(*zip(*km2.cluster_centers_), c='r')
plt.title("Prédiction pour k = 2")

# k = 3
km3 = KMeans(3).fit(data)
plt.subplot(223)
plt.scatter(x, y, c=km3.labels_, alpha=0.7)
plt.scatter(*zip(*km3.cluster_centers_), c='r')
plt.title("Prédiction pour k = 3")

# k = 4
km4 = KMeans(4).fit(data)
plt.subplot(224)
plt.scatter(x, y, c=km4.labels_, alpha=0.7)
plt.scatter(*zip(*km4.cluster_centers_), c='r')
plt.title("Prédiction pour k = 4")

plt.tight_layout()
plt.show()