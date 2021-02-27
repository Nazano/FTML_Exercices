# Reprenez les courbes du premier exercice et annotez-les pour
# distinguer les points y = exp(1).

import matplotlib.pyplot as plt
import numpy as np

lambda_val = [1,5,10]
x = np.linspace(0, 1)

for val in lambda_val:
    exposant = -val * x
    plt.plot(x, np.exp(exposant), label=f"lambda: {val}")
    plt.annotate('y=1/e', (1/val, np.exp(-1)))

plt.xlabel("x")
plt.ylabel("y=exp(-lambda x)")
plt.legend()
plt.show()
