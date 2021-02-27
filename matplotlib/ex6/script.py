import pandas as pd
import matplotlib.pyplot as plt
from os import path

df = pd.read_csv(path.join(path.dirname(__file__), path.normpath('../../data/data1.csv')))

colors = df['survived'][df['fare'].notna() & df['age'].notna()].apply(lambda x: 'red' if x == 0 else 'green') # Colorie seulement quand fare et age non nul
pd.plotting.scatter_matrix(df[['fare', 'age']], diagonal='kde', c=colors)
plt.show()