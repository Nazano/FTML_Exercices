# Importez les données de data1.csv dans un dataframe et
# affichez leur dispersion selon les axes age et prix du billet.

import pandas as pd
import matplotlib.pyplot as plt
from os import path

df = pd.read_csv(path.join(path.dirname(__file__), path.normpath('../../data/data1.csv')))

plt.subplot(211)
plt.scatter(df['age'], df['fare'])
plt.xlabel('age')
plt.ylabel('tarif')

plt.subplot(212)
plt.scatter(df['age'][df['survived'] == 1], df['fare'][df['survived'] == 1], c='green', label='survived')
plt.scatter(df['age'][df['survived'] == 0], df['fare'][df['survived'] == 0], c='yellow', label='no')
plt.xlabel('age')
plt.ylabel('tarif')

plt.tight_layout()
plt.legend()
plt.show()