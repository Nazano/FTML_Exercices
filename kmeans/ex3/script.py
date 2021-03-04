# Importez les données iris. Effectuez une classification
# hiérarchique ascendante avec la distance de Ward. Fixez un
# seuil optimal ou un nombre de compsantes. Représentez
# graphiquement les clusters et comparez avec les valeurs de Y.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

iris = load_iris()
data = np.delete(iris.data, slice(-2, None), 1) # On garde que les deux premières colonnes
x, y = zip(*data)

plt.subplot(2,2,1)
plt.scatter(x, y, c=iris.target, alpha=0.7)
plt.title("Données d'apprentissage")

for k in range(2,5):
    ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
    ward.fit(np.vstack((x, y)).T)
    plt.subplot(2,2,k)
    plt.scatter(x, y, c=ward.labels_, alpha=0.7)
    plt.title(f"Prédiction pour k = {k}")

plt.tight_layout()
plt.show()