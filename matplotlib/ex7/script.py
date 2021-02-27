import pandas as pd
import matplotlib.pyplot as plt
from os import path

df = pd.read_csv(path.join(path.dirname(__file__), path.normpath('../../data/data1.csv')))

df = df.dropna(subset=['age', 'fare', 'sex'])
df['sex'] = df['sex'].apply(lambda x: {'male' : 0, 'female' : 1}[x])

colors = list(map(lambda x: 'red' if x == 0 else 'green', df['survived']))

pd.plotting.scatter_matrix(df[['fare', 'sex', 'age']], diagonal='hist', c=colors)
plt.show()