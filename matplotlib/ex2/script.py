# Même question, mais les courbes doivent être sur des
# sous-figures différentes et non superposées.

import matplotlib.pyplot as plt
import numpy as np

lambda_val = [1,5,10]
x = np.linspace(0, 1)

pos = 1
for val in lambda_val:
    plt.subplot(2,2,pos)
    plt.plot(np.exp(-val * x), x, label=f"y=exp(-{val}x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    pos += 1

plt.tight_layout()
plt.show()