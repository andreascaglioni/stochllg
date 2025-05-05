import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("./")
from stochllg.parametric_W import param_LC_W


yy = np.random.normal(0, 1, (5000, 10000))
tt = np.linspace(0, 1, num=1000)
W = param_LC_W(yy, tt, 1)

# Moments
mean = np.mean(W, axis=0)
var = np.var(W, axis=0)
print("Var(W(1)) = ", var[-1])
print("E(W(0)) = ", mean[-1])

# Plot
plt.figure()
plt.plot(tt, W[:10,].T)  # NB plots COLUMNS!

plt.figure()
plt.plot(tt, mean, label='mean')
plt.plot(tt, var, label='variance')
plt.plot(tt, 0*tt, "k--", label='theor. mean')
plt.plot(tt, tt, "k-", label='theor. var.')
plt.legend()

plt.show()