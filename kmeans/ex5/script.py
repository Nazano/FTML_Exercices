# Même chose, mais commencez par normaliser les données
# avec MinMaxScaler.

import pandas as pd
import matplotlib.pyplot as plt
from os import path
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(path.join(path.dirname(__file__), path.normpath('../../data/data1.csv')))
df = df[['age', 'fare']].dropna()

scaler = MinMaxScaler().fit(df)
df = pd.DataFrame(scaler.transform(df), columns=['age', 'fare'])

for k in range(4,6):
    # KMEANS
    km = KMeans(n_clusters=k).fit(df)
    plt.subplot(2,2, k - 3)
    plt.scatter(df['age'], df['fare'], c=km.labels_, alpha=0.8)
    plt.scatter(*zip(*km.cluster_centers_), c='r')
    plt.xlabel('age')
    plt.ylabel('fare')
    plt.title(f"Kmeans k={k}")
    
    # CAH
    cah = AgglomerativeClustering(n_clusters=k).fit(df)
    plt.subplot(2,2, k - 1)
    plt.scatter(df['age'], df['fare'], c=cah.labels_, alpha=0.8)
    plt.title(f"CAH k={k}")
    plt.xlabel('age')
    plt.ylabel('fare')

plt.tight_layout()
plt.show()