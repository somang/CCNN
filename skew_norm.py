from scipy.stats import skewnorm
import matplotlib.pyplot as plt
import numpy as np

DATASIZE = 1000
SKEW_PARAMS = [-3, 0]

def randn_skew(N, alpha=0.0, loc=0.0, scale=1.0):
    sigma = alpha / np.sqrt(1.0 + alpha**2) 
    u0 = np.random.randn(N)
    v = np.random.randn(N)
    u1 = (sigma*u0 + np.sqrt(1.0 - sigma**2)*v) * scale
    u1[u0 < 0] *= -1
    u1 = u1 + loc
    return u1

# lets check again
p = -1 * randn_skew(DATASIZE, -100, 120, 500)
print(np.rint(p))
print(max(np.rint(p)), min(np.rint(p)), np.average(p))
plt.plot(p)
plt.show()
