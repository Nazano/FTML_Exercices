# Affichez les fonctions x -> exp(-λx) pour diverses valeurs du
# paramètre λ. N’oubliez pas l’axe et la légende.

import matplotlib.pyplot as plt
import numpy as np

lambda_val = [1,5,10]
x = np.linspace(0, 1)

for val in lambda_val:
    plt.plot(x, np.exp(-val * x), label=f"lambda: {val}")

plt.legend()
plt.show()