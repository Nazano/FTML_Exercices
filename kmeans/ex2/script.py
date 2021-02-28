# Séparez maintenant vos données en un échantillon
# d’apprentissage et un échantillon de test. Visualisez (par
# exemple avec des différences d’opacité) dans quels clusters
# sont placées les nouvelles valeurs.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

iris = load_iris()
data = np.delete(iris.data, slice(-2, None), 1) # On garde que les deux premières colonnes
x, y = zip(*data)
x_train, x_test, y_train, y_test = train_test_split(np.vstack((x, iris.target)).T, np.vstack((y, iris.target)).T, test_size=0.5)

# Récupère les labels
x_train, train_labels = zip(*x_train)
x_test, test_labels = zip(*x_test)
y_train, _ = zip(*y_train)
y_test, _ = zip(*y_test)

plt.subplot(2,2,1)
plt.scatter(x_train, y_train, c=train_labels, alpha=0.7)
plt.title("Données d'apprentissage")

for k in range(2,5):
    km = KMeans(k).fit(np.vstack((x_train, y_train)).T)
    y_pred_test_labels = km.predict(np.vstack((x_test, y_test)).T)
    plt.subplot(2,2,k)
    plt.scatter(x_train, y_train, c=km.labels_, alpha=0.7) 
    plt.scatter(x_test, y_test, c=y_pred_test_labels,alpha=0.7)
    plt.scatter(*zip(*km.cluster_centers_), c='r')
    plt.title(f"Prédiction pour k = {k}")

plt.tight_layout()
plt.show()